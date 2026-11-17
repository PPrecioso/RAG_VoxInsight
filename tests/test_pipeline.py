from pathlib import Path
import pandas as pd
from voc_rag.data import clean_text, build_documents
from voc_rag.rag import answer_question, sufficient, construct_context


def test_clean_text():
    assert clean_text(" Entrega\n  ruim\t") == "Entrega ruim"
    assert clean_text(float("nan")) == ""


def test_build_documents_no_duplication(tmp_path):
    pd.DataFrame([
        {"review_id": "r1", "order_id": "o1", "review_score": 1, "review_comment_title": "Atrasou", "review_comment_message": "Produto chegou depois do prazo"},
        {"review_id": "r2", "order_id": "o2", "review_score": 5, "review_comment_title": None, "review_comment_message": "Ótimo atendimento"},
        {"review_id": "r3", "order_id": "o3", "review_score": 3, "review_comment_title": None, "review_comment_message": None},
    ]).to_csv(tmp_path / "olist_order_reviews_dataset.csv", index=False)
    pd.DataFrame([
        {"order_id": "o1", "product_id": "p1"}, {"order_id": "o1", "product_id": "p2"}
    ]).to_csv(tmp_path / "olist_order_items_dataset.csv", index=False)
    pd.DataFrame([
        {"product_id": "p1", "product_category_name": "beleza"},
        {"product_id": "p2", "product_category_name": "livros"}
    ]).to_csv(tmp_path / "olist_products_dataset.csv", index=False)
    docs = build_documents(tmp_path)
    assert len(docs) == 2
    assert docs.review_id.is_unique
    assert docs.loc[docs.review_id == "r1", "categories"].iloc[0] == "beleza | livros"


def test_refuses_out_of_scope():
    class Index:
        def retrieve(self, *args, **kwargs):
            raise AssertionError("Não deve recuperar perguntas fora de escopo")
    result = answer_question(Index(), "Qual a previsão do tempo em São Paulo?")
    assert result["mode"] == "abstention"


def test_refuses_weak_evidence():
    assert not sufficient("Entrega?", [{"similarity": 0.02, "relevance": 0.02}])


def test_extracts_real_ids():
    class Index:
        def retrieve(self, *args, **kwargs):
            return [{"review_id": "r1", "order_id": "o1", "review_score": 1,
                     "text": "A entrega demorou muito", "similarity": .8, "relevance": .8}]
    result = answer_question(Index(), "O que houve na entrega?", client=None)
    assert "r1" in result["answer"]
    assert "r1" in construct_context(result["evidence"])


def test_missing_metadata_in_context():
    m = {"review_id": "r1", "review_score": 2, "text": "Demora na entrega", "categories": pd.NA,
         "delivery_delay_days": float("nan"), "review_date": pd.NA}
    context = construct_context([m])
    assert "não informada" in context
    assert "atraso_dias" not in context
