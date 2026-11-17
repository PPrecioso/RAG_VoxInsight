"""Ingestão, limpeza, enriquecimento e preparação da base Olist."""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

REVIEWS = "olist_order_reviews_dataset.csv"
ORDERS = "olist_orders_dataset.csv"
ITEMS = "olist_order_items_dataset.csv"
PRODUCTS = "olist_products_dataset.csv"
CATEGORIES = "product_category_name_translation.csv"


def download_dataset(directory: str | Path) -> Path:
    """Baixa o dataset oficial pelo KaggleHub (requer internet na execução)."""
    import kagglehub
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = Path(kagglehub.dataset_download("olistbr/brazilian-ecommerce"))
    import shutil
    names = [REVIEWS, ORDERS, ITEMS, PRODUCTS, CATEGORIES]
    for name in names:
        found = list(source.rglob(name))
        if found:
            shutil.copyfile(found[0], directory / name)
    if not (directory / REVIEWS).exists():
        raise FileNotFoundError("Arquivo de avaliações não encontrado no download.")
    return directory


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    txt = str(value)
    txt = re.sub(r"[\x00-\x1f\x7f]+", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def build_documents(directory: str | Path) -> pd.DataFrame:
    """1 review_id = 1 documento; metadados opcionais não duplicam avaliações."""
    directory = Path(directory)
    file = directory / REVIEWS
    if not file.exists():
        raise FileNotFoundError(f"Não encontrei {file}; execute: python -m voc_rag.cli download")
    df = pd.read_csv(file, dtype={"review_id": "string", "order_id": "string"})
    required = {"review_id", "order_id", "review_score", "review_comment_title", "review_comment_message"}
    if not required.issubset(df.columns):
        raise ValueError(f"Faltam colunas obrigatórias: {required - set(df.columns)}")
    df = df.drop_duplicates(subset=["review_id"], keep="first").copy()
    df["title"] = df["review_comment_title"].map(clean_text)
    df["comment"] = df["review_comment_message"].map(clean_text)
    df["text"] = (df["title"] + ". " + df["comment"]).str.strip(" .")
    # Avaliações só com nota, sem título ou texto, não sustentam respostas semânticas.
    df = df[df["text"].str.len() >= 4].copy()
    df["review_score"] = pd.to_numeric(df["review_score"], errors="coerce")
    df = df[df["review_score"].between(1, 5)].copy()
    if "review_creation_date" in df.columns:
        df["review_date"] = pd.to_datetime(df["review_creation_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        df["review_date"] = None

    orders_file = directory / ORDERS
    if orders_file.exists():
        cols = ["order_id", "order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date", "order_status"]
        od = pd.read_csv(orders_file, usecols=lambda c: c in cols, dtype={"order_id": "string"}).drop_duplicates("order_id")
        for name in ["order_delivered_customer_date", "order_estimated_delivery_date"]:
            if name in od:
                od[name] = pd.to_datetime(od[name], errors="coerce")
        if {"order_delivered_customer_date", "order_estimated_delivery_date"}.issubset(od.columns):
            delta = od["order_delivered_customer_date"] - od["order_estimated_delivery_date"]
            od["delivery_delay_days"] = delta.dt.total_seconds() / 86400
        df = df.merge(od, on="order_id", how="left", validate="many_to_one")

    # Categoria por pedido: agrupa várias categorias sem multiplicar linhas.
    if (directory / ITEMS).exists() and (directory / PRODUCTS).exists():
        items = pd.read_csv(directory / ITEMS, usecols=["order_id", "product_id"], dtype="string")
        products = pd.read_csv(directory / PRODUCTS, usecols=["product_id", "product_category_name"], dtype="string")
        items = items.merge(products.drop_duplicates("product_id"), on="product_id", how="left", validate="many_to_one")
        if (directory / CATEGORIES).exists():
            names = pd.read_csv(directory / CATEGORIES, dtype="string")
            items = items.merge(names.drop_duplicates("product_category_name"), on="product_category_name", how="left", validate="many_to_one")
            cat_col = "product_category_name_english"
        else:
            cat_col = "product_category_name"
        cats = (items.dropna(subset=[cat_col]).groupby("order_id")[cat_col]
                .agg(lambda x: " | ".join(sorted(set(x.astype(str))))).rename("categories"))
        df = df.join(cats, on="order_id")
    keep = ["review_id", "order_id", "review_score", "review_date", "text", "title", "comment",
            "categories", "delivery_delay_days", "order_status", "order_purchase_timestamp"]
    for col in keep:
        if col not in df:
            df[col] = None
    return df[keep].reset_index(drop=True)
