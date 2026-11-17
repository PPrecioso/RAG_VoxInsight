# Roteiro de vídeo — Tech Challenge Fase 4 (até 7 minutos)

**0:00–0:45 | Contexto de negócio.** Gestores precisam explorar avaliações da Olist em linguagem natural, além da nota numérica.

**0:45–1:30 | Base e tratamento.** Mostre o dataset, a preparação de comentários, os valores ausentes, a deduplicação por `review_id`, os 41.461 registros (se a contagem for confirmada na sua execução) e os metadados opcionais.

**1:30–2:30 | Busca e arquitetura.** Explique TF-IDF, embeddings densos LSA e pontuação híbrida `0,65 × LSA + 0,35 × TF-IDF`, filtros por nota/categoria, seleção top-k e construção de contexto com fontes verificáveis. Deixe claro que LSA é alternativa eficiente a embeddings neurais, não equivalente.

**2:30–3:20 | Geração REAL e segurança.** Na sidebar selecione `Ollama · IA local` (ou OpenAI). Mostre o modelo instalado. Explique que a consulta é enviada ao LLM junto com os comentários recuperados; a resposta exige citações por `review_id`. Mostre o indicador `generativo`. Diga que validação de IDs não é garantia de *faithfulness* perfeita.

**3:20–5:30 | Demonstração.** Pergunte por reclamações de entrega, mostre a síntese interpretativa e abra as avaliações citadas. Filtre nota 1–2 e consulte insatisfação. Mude para notas 4–5 e pergunte elogios. Mostre um gráfico do panorama. **Use apenas respostas reais que sua aplicação produzir.**

**5:30–6:10 | Evidências insuficientes.** Pergunte sobre previsão do tempo, mostre a abstenção. Explique como a interface reage ao LLM indisponível.

**6:10–7:00 | Decisões e limites.** Discuta performance local, custo zero de API com Ollama, limitações do LSA, top-k e limiares. Cite próximos passos: métricas Recall@k, avaliação humana de fidelidade, melhoria de embeddings e testes de robustez.

## Checklist antes de gravar
- [ ] Ollama instalado, modelo disponível em `ollama list` (ou API OpenAI válida)
- [ ] `data/raw` e `data/processed` preparados
- [ ] `python -m pytest -q` passando
- [ ] Resposta realmente indica `Modo de resposta: generativo`
- [ ] Comentários citados existem na seção de evidências
- [ ] Nenhuma chave de API exibida
- [ ] Repositório GitHub público publicado
- [ ] Vídeo com até 7 minutos gravado e enviado
