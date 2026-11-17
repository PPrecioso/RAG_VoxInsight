"""Voice of Customer Intelligence · painel executivo Streamlit."""
from __future__ import annotations
import os
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from voc_rag.index import ReviewIndex
from voc_rag.rag import answer_question
from voc_rag.llm import check_ollama
from voc_rag.insights import theme_counts, themes_for

load_dotenv(ROOT / ".env")
st.set_page_config(page_title="VoxInsight • Olist", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"],.stApp{font-family:'DM Sans',sans-serif;}
.stApp{background:radial-gradient(ellipse at 72% 0%,rgba(41,108,135,.20),transparent 46%),#091521;}
.block-container{padding-top:2rem;max-width:1290px;padding-bottom:5rem;}
[data-testid="stSidebar"]{background:#0D2030;border-right:1px solid #213b4d;}
[data-testid="stSidebar"] .block-container{padding-top:2rem;}
h1,h2,h3{font-family:Manrope,sans-serif!important;letter-spacing:-.04em!important;}
h1{font-size:2.8rem!important;font-weight:800!important;}
h2{font-size:1.55rem!important;font-weight:750!important;}
h3{font-size:1.12rem!important;font-weight:700!important;}
[data-testid="stMetric"]{background:#102638;border:1px solid #244053;padding:18px 20px;border-radius:16px;min-height:120px;}
[data-testid="stMetricLabel"]{color:#99B0C0!important;}
[data-testid="stMetricValue"]{color:#E9F9FF!important;font-family:Manrope;font-weight:800;}
[data-testid="stSelectbox"]>div>div,[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea{background:#122B3D!important;border-color:#2A4858!important;border-radius:12px!important;}
.stButton>button[kind="primary"]{background:#2ED5BB;color:#06242F;border:0;border-radius:12px;padding:0.65rem 1.45rem;font-weight:800;}
.stButton>button[kind="primary"]:hover{background:#5BE3CF;color:#06242F;}
[data-testid="stExpander"]{border:1px solid #234153!important;border-radius:14px!important;background:#112638;}
[data-testid="stTabs"] button{font-weight:700;}
hr{border-color:#244052!important;}
.hero{border:1px solid #295464;border-radius:22px;background:linear-gradient(115deg,#133D4C,#112638 56%,#15243D);padding:35px 39px;margin-bottom:24px;position:relative;overflow:hidden;}
.eyebrow{font-size:11px;letter-spacing:.17em;font-weight:800;color:#59DCCA;margin-bottom:12px;}
.hero h1{margin:0!important;color:#F2FBFF;line-height:1.14;font-size:2.6rem!important;}
.hero p{font-size:15px;color:#ABC6D2;max-width:720px;margin:16px 0 0;line-height:1.65;}
.chip{display:inline-block;border:1px solid #3A6570;background:#174852;color:#8DF1DE;font-size:11px;padding:5px 10px;border-radius:30px;margin-top:17px;margin-right:7px;font-weight:700;}
.section-label{font-size:11px;color:#64D9C7;letter-spacing:.14em;font-weight:800;margin:24px 0 9px;}
.intro-note{font-size:13px;color:#8DA9BB;}
.answer-panel{border:1px solid #2A5960;background:linear-gradient(115deg,#14353B,#112A39);padding:8px 21px;border-radius:16px;}
.brand{font-size:21px;font-weight:800;letter-spacing:-.05em;color:#F2F9FF;margin:0 0 5px;}
.brand span{color:#2ED5BB;}
.caption-small{font-size:12px;color:#8EABBC;}
</style>""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Carregando base de conhecimento...")
def get_index():
    return ReviewIndex(ROOT / "data" / "processed")

try:
    index = get_index()
except (FileNotFoundError, OSError, ValueError) as exc:
    st.error(f"Não foi possível carregar o índice: {exc}")
    st.code("python -m voc_rag.cli download\npython -m voc_rag.cli build --backend lite\npython -m streamlit run app.py", language="powershell")
    st.stop()

with st.sidebar:
    st.markdown('<div class="brand">◈ Vox<span>Insight</span></div><div class="caption-small">Customer Intelligence / Olist</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown('### Filtros de exploração')
    score_filter = st.selectbox("Perfil de avaliação", ["Todas as notas", "Insatisfeitos · 1–2", "Neutros · 3", "Satisfeitos · 4–5"])
    category_names = sorted(set(" | ".join(index.df["categories"].dropna().astype(str)).split(" | ")) - {""})
    selected_category = st.selectbox("Categoria do produto", ["Todas as categorias"] + category_names)
    k = st.slider("Evidências na resposta", 3, 15, 8)
    st.divider()
    st.markdown("### IA generativa")
    provider_label = st.radio("Motor de respostas", ["Ollama · IA local (grátis)", "OpenAI · API", "Extrativo · sem LLM"], index=0, help="O modo Ollama produz texto generativo real, usando um modelo instalado no computador.")
    provider = "ollama" if provider_label.startswith("Ollama") else "openai" if provider_label.startswith("OpenAI") else "extrativo"
    if provider == "ollama":
        available = check_ollama()
        llm_model = st.selectbox("Modelo local", available + ([os.getenv('OLLAMA_MODEL','gemma3:4b')] if os.getenv('OLLAMA_MODEL','gemma3:4b') not in available else []))
        if available:
            st.success("Ollama conectado. Modelos encontrados: " + ", ".join(available[:4]))
        else:
            st.warning("Ollama não encontrado ou sem modelos. Instale Ollama e execute no terminal: ollama pull gemma3:4b")
        api_key = None
    elif provider == "openai":
        llm_model = st.text_input("Modelo OpenAI", value=os.getenv('OPENAI_MODEL','gpt-4.1-mini'))
        api_key = st.text_input("Chave da API (opcional se existir no .env)", type="password", help="Nunca publique a chave no GitHub.") or None
        if not (api_key or os.getenv('OPENAI_API_KEY')):
            st.warning("Configure sua chave no .env ou neste campo. A API pode ter custo.")
    else:
        llm_model, api_key = "", None
    st.divider()
    st.caption("BASE DE CONHECIMENTO")
    st.metric("Avaliações indexadas", f"{len(index.df):,}".replace(",", "."))
    st.caption(f"Busca: {'Embeddings neurais' if index.backend == 'transformer' else 'LSA + TF-IDF'}")
    st.caption(f"Motor selecionado: {provider.upper()}")
    st.info("A classificação de temas é aproximada. Os resultados da busca não representam uma pesquisa estatística da base completa.")

limits={"Todas as notas":(None,None),"Insatisfeitos · 1–2":(1,2),"Neutros · 3":(3,3),"Satisfeitos · 4–5":(4,5)}
lo,hi=limits[score_filter]
filtered = index.df
if lo is not None:
    filtered=filtered[filtered["review_score"].between(lo,hi)]
if selected_category != "Todas as categorias":
    filtered=filtered[filtered["categories"].fillna("").str.contains(selected_category,case=False,regex=False)]

st.markdown("""<div class="hero"><div class="eyebrow">BUSINESS INTELLIGENCE  /  VOICE OF CUSTOMER</div>
<h1>Entenda o que seus clientes<br>realmente estão dizendo.</h1>
<p>Explore milhares de avaliações da Olist com recuperação híbrida, filtros de contexto e evidências rastreáveis para apoiar decisões mais informadas.</p>
<span class="chip">◈ RAG rastreável</span><span class="chip">◇ Dados reais da Olist</span><span class="chip">✓ IA local com Ollama</span></div>""",unsafe_allow_html=True)

m1,m2,m3,m4=st.columns(4)
with m1: st.metric("Avaliações no filtro", f"{len(filtered):,}".replace(",","."))
with m2: st.metric("Nota média", f"{filtered['review_score'].mean():.2f}".replace(".",",") if len(filtered) else "—")
with m3: st.metric("Notas 1–2", f"{(filtered['review_score'].le(2).mean()*100):.1f}%".replace(".",",") if len(filtered) else "—")
with m4: st.metric("Categorias", f"{len(category_names)}")

st.markdown('<div class="section-label">WORKSPACE / EXPLORAR A VOZ DO CLIENTE</div>', unsafe_allow_html=True)
tab1,tab2,tab3=st.tabs(["◈  Perguntar à base", "▥  Panorama das avaliações", "◎  Metodologia e limites"])
with tab1:
    st.write("")
    examples=["Quais reclamações aparecem sobre o prazo de entrega?", "O que os compradores elogiam sobre a experiência de compra?", "Quais problemas de qualidade são relatados?", "Quais são as principais reclamações nas avaliações de nota baixa?", "Que relatos mencionam pedidos incorretos ou incompletos?"]
    suggestion=st.selectbox("Perguntas sugeridas",["Escolha uma sugestão ou escreva abaixo"]+examples)
    if suggestion != "Escolha uma sugestão ou escreva abaixo" and suggestion != st.session_state.get("last_suggestion"):
        st.session_state["draft_question"]=suggestion
    st.session_state["last_suggestion"]=suggestion
    question=st.text_area("Sua pergunta", key="draft_question", placeholder="Ex.: Quais são os motivos de insatisfação relacionados à entrega?",height=100)
    run=st.button("Analisar avaliações  →",type="primary",use_container_width=False)
    if run:
        if not question.strip():
            st.warning("Digite uma pergunta para iniciar a análise.")
        else:
            with st.spinner("Recuperando avaliações relevantes e construindo a resposta..."):
                st.session_state["last_result"]=answer_question(index,question,k=k,min_score=lo,max_score=hi,category=None if selected_category=="Todas as categorias" else selected_category,provider=provider,model=llm_model,api_key=api_key)
                st.session_state["last_query"]=question
                st.session_state["last_filters"]=(score_filter,selected_category)
    result=st.session_state.get("last_result")
    if result:
        st.markdown('<div class="section-label">RESULTADO / RESPOSTA FUNDAMENTADA</div>',unsafe_allow_html=True)
        st.markdown("### Síntese dos relatos")
        st.markdown(result['answer'])
        if result.get('warning'): st.warning(result['warning'])
        if result['mode']=='generativo': st.success("Resposta gerada por um modelo de linguagem com IDs de avaliações verificados.")
        if result['mode']=='llm_error': st.error("O RAG recuperou os dados, mas a etapa generativa falhou. Verifique as instruções de instalação do Ollama ou da API.")
        if result['mode']=='abstention': st.warning("Não foi possível sustentar uma resposta com as evidências recuperadas.")
        st.caption(f"Modo de resposta: {result['mode']} · {len(result['evidence'])} evidências recuperadas · filtros da busca: {st.session_state.get('last_filters')}")
        st.markdown('### Evidências verificáveis')
        st.caption("Comentários originais, identificados por review_id. A similaridade é uma pontuação de recuperação, não confiança probabilística.")
        for i,m in enumerate(result['evidence'],1):
            with st.expander(f"{i:02d} · Nota {int(m['review_score'])}/5  ·  Relevância {m['relevance']:.2f}  ·  {str(m['review_id'])[:12]}…",expanded=(i==1)):
                st.write(str(m['text']))
                st.caption(f"Review ID: {m['review_id']} | Pedido: {m.get('order_id')} | Data: {m.get('review_date')} | Categoria: {m.get('categories')}")
                themes=themes_for(m.get('text',''))
                if themes: st.caption("Temas reconhecidos: " + ", ".join(themes))
        if result['evidence']:
            st.download_button("Exportar evidências em CSV",pd.DataFrame(result['evidence']).to_csv(index=False).encode('utf-8-sig'),file_name="evidencias_olist.csv",mime="text/csv")
    else:
        st.info("Selecione uma pergunta e clique em **Analisar avaliações**. Aqui aparecerão a resposta e suas fontes.")
with tab2:
    st.write("")
    left,right=st.columns([1,1.2],gap="large")
    with left:
        st.markdown("### Distribuição de notas")
        dist=filtered['review_score'].value_counts().reindex([1,2,3,4,5],fill_value=0).rename_axis('Nota').reset_index(name='Avaliações')
        st.bar_chart(dist,x='Nota',y='Avaliações',color='#2ED5BB',height=310)
    with right:
        st.markdown("### Menções textuais por tema")
        topics=theme_counts(filtered)
        st.bar_chart(topics,x='Tema',y='Avaliações com menção',color='#58A9FF',height=310)
    st.caption("Contagens por palavras-chave em TODAS as avaliações do filtro; temas não são mutuamente exclusivos e menção não implica reclamação ou elogio. Nenhuma classificação de sentimento foi inferida.")
    st.dataframe(topics,hide_index=True,use_container_width=True)
with tab3:
    st.markdown("### Como funciona o pipeline")
    st.markdown("**1. Ingestão:** limpeza/deduplicação por review_id.  \n**2. Embeddings:** LSA local, com modo neural opcional.  \n**3. Recuperação:** busca híbrida e filtros.  \n**4. Contexto:** avaliações e metadados.  \n**5. Geração:** Ollama local ou OpenAI produzem sínteses interpretativas.  \n**6. Verificação:** IDs citados precisam existir no conjunto recuperado; abstenção se não houver evidências.")
    st.markdown("### Limites metodológicos")
    st.warning("O modo local identifica temas com regras de palavras-chave, não compreende plenamente ironia, negação ou contexto. Para inferências qualitativas mais sofisticadas, configure o LLM e valide manualmente as respostas.")
    st.markdown("**Importante:** percentuais e distribuição de notas são calculados na base textual filtrada, não em todos os pedidos do dataset. Rankings por tema são contagens de menções, não de problemas. Não extrapole os resultados dos k comentários recuperados para a população inteira.")
    st.markdown("**Fonte:** [Brazilian E-Commerce Public Dataset by Olist (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data)")
