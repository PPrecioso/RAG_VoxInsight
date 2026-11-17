import pandas as pd
from voc_rag.insights import theme_counts, themes_for, summarize_evidence

def test_theme_normalization():
    assert "Entrega e prazo" in themes_for("A ENTREGA atrasou muito")
    assert "Produto e qualidade" in themes_for("Produto quebrado")

def test_counts_are_nonexclusive():
    df=pd.DataFrame({"text":["Entrega atrasada e produto quebrado","Atendimento bom"]})
    values=theme_counts(df).set_index("Tema")["Avaliações com menção"]
    assert values["Entrega e prazo"]==1
    assert values["Produto e qualidade"]==1

def test_local_summary_is_cited_and_calibrated():
    matches=[{"review_id":"r123", "text":"A entrega atrasou muito"}]
    answer=summarize_evidence(matches,"Problemas na entrega")
    assert "[r123]" in answer
    assert "amostra de busca" in answer
