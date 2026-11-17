from voc_rag.llm import LLMError, OllamaGenerator, select_generator
from voc_rag.rag import answer_question, validate_citations, construct_context


class Index:
    def retrieve(self, *args, **kwargs):
        return [dict(review_id="r1", text="O pedido atrasou dez dias", review_score=1,
                     categories="livros", similarity=.8, relevance=.8),
                dict(review_id="r2", text="O pacote chegou danificado", review_score=2,
                     categories="livros", similarity=.7, relevance=.7)]


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def generate(self, system, user):
        self.calls.append((system, user))
        return self.reply


def test_generates_interpretative_text_with_real_citations():
    llm = FakeLLM("Os relatos sugerem problemas de prazo [r1] e de embalagem [r2].")
    result = answer_question(Index(), "O que preocupa os compradores?", provider="ollama", client=llm)
    assert result['mode'] == 'generativo'
    assert 'problemas de prazo' in result['answer']
    assert len(llm.calls) == 1
    assert '[r1]' in llm.calls[0][1]


def test_hallucinated_citation_is_rejected():
    llm = FakeLLM("Muitos consumidores relatam fraude [id_inventado].")
    result = answer_question(Index(), "Quais problemas foram identificados?", client=llm)
    assert result['mode'] == 'abstention_llm'
    assert 'id_inventado' not in result['answer']


def test_no_citations_is_rejected():
    llm = FakeLLM("O atraso é frequente.")
    result = answer_question(Index(), "O que dizem sobre entrega?", client=llm)
    assert result['mode'] == 'abstention_llm'


def test_model_failure_is_explicit():
    class Down:
        def generate(self, system, user):
            raise LLMError('Ollama offline')
    result = answer_question(Index(), "O que dizem sobre entrega?", client=Down())
    assert result['mode'] == 'llm_error'
    assert 'Ollama offline' in result['warning']


def test_extractive_fallback_remains_user_selected():
    result = answer_question(Index(), "O que dizem sobre entrega?", provider='extrativo')
    assert result['mode'] == 'extrativo'


def test_select_local_generator():
    assert isinstance(select_generator('ollama', 'gemma3:1b'), OllamaGenerator)
    assert select_generator('extrativo') is None


def test_context_contains_ids():
    assert 'ID=[r1]' in construct_context(Index().retrieve())
