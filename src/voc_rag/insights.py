"""Análise descritiva interpretável, separada de evidências RAG."""
from __future__ import annotations
import re
import unicodedata
from collections import Counter
import pandas as pd

THEMES = {
    "Entrega e prazo": r"\b(entreg\w*|atras\w*|prazo\w*|transport\w*|frete\w*|cheg\w*|receb\w*)\b",
    "Produto e qualidade": r"\b(defeit\w*|quebrad\w*|qualidad\w*|danific\w*|avari\w*|estragad\w*|funcion\w*|tamanho\w*|peca\w*)\b",
    "Atendimento": r"\b(atend\w*|suport\w*|reclam\w*|respost\w*|contato\w*|sac)\b",
    "Pedido incorreto": r"\b(errad\w*|trocad\w*|diferent\w*|falt\w*|incomplet\w*|outro produto|nao veio)\b",
    "Preço e pagamento": r"\b(preco\w*|pagament\w*|cobranc\w*|reembols\w*|dinheiro|caro\w*|valor\w*)\b",
    "Embalagem": r"\b(embalag\w*|caixa\w*|pacote\w*|prote\w*)\b",
}

def normalize_text(text: object) -> str:
    text = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"\s+", " ", text)

def themes_for(text: object) -> list[str]:
    normalized = normalize_text(text)
    return [name for name, pattern in THEMES.items() if re.search(pattern, normalized)]

def theme_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Contagem não exclusiva de MENÇÕES textuais por tema, não sentimento nem incidência de problemas."""
    counts = Counter()
    for text in df["text"].fillna(""):
        counts.update(themes_for(text))
    return pd.DataFrame([{"Tema": k, "Avaliações com menção": counts[k]} for k in THEMES]).sort_values("Avaliações com menção", ascending=False).reset_index(drop=True)

def summarize_evidence(matches: list[dict], question: str) -> str:
    """Síntese EXTRATIVA e descritiva: nenhum insight sem ID da avaliação."""
    if not matches:
        return "Não há evidências suficientes para esta pergunta com os filtros selecionados."
    cnt = Counter()
    for m in matches:
        cnt.update(themes_for(m.get("text", "")))
    themes = cnt.most_common(3)
    intro = (f"Foram recuperadas **{len(matches)} avaliações** relevantes para a pergunta. "
             "Esta é uma **amostra de busca**, não uma medida de frequência de toda a base. ")
    if themes:
        details = []
        for theme, n in themes:
            proof = [str(m["review_id"]) for m in matches if theme in themes_for(m.get("text", ""))][:2]
            details.append(f"**{theme}** ({n} de {len(matches)} avaliações recuperadas; exemplos: " + ", ".join(f"[{p}]" for p in proof) + ")")
        intro += "Os temas identificados por palavras-chave neste conjunto são: " + "; ".join(details) + "."
    else:
        ids = ", ".join(f"[{m['review_id']}]" for m in matches[:3])
        intro += f"Não foi possível classificar temas com as regras disponíveis; consulte as evidências {ids}."
    return intro + "\n\n*Classificação aproximada por termos; a interpretação deve ser conferida nos relatos abaixo.*"
