# Como obter respostas reais de IA generativa — Windows

## A opção gratuita, sem chave de API: Ollama

1. Instale o **Ollama para Windows** a partir de https://ollama.com/download/windows .
2. Abra o PowerShell (pode ser o terminal do VS Code) e execute:
   ```powershell
   ollama pull gemma3:4b
   ```
   É um download adicional do modelo e **não está dentro do ZIP**. Necessita internet, espaço em disco e memória compatíveis.
   Em computadores modestos, tente `ollama pull gemma3:1b` e escolha `gemma3:1b` na interface.
3. Deixe o aplicativo Ollama em execução. Se não houver servidor ativo, inicie outro terminal com `ollama serve`.
4. Inicie o VoxInsight com `powershell -ExecutionPolicy Bypass -File .\INICIAR_WINDOWS.ps1` ou pelos comandos do README.
5. No menu lateral, em **IA generativa**, selecione **Ollama · IA local (grátis)** e escolha o modelo baixado.
6. Faça uma pergunta e clique em **Analisar avaliações**. O indicador deve mostrar `Modo de resposta: generativo`.

Teste de verificação:
```powershell
ollama list
Invoke-RestMethod http://127.0.0.1:11434/api/tags
```

## Alternativa na nuvem: OpenAI

1. Configure uma chave de API válida em `.env` (copie `.env.example` para `.env`).
2. Nunca envie a chave ao GitHub nem tire print contendo a chave.
3. Selecione **OpenAI · API** na barra lateral e informe o modelo.
4. A **API tem cobrança separada do ChatGPT**; o uso requer créditos/conta de API habilitada.

## O que significa "generativo" neste projeto

O pipeline recupera até `k` comentários semanticamente relacionados, envia o contexto ao LLM e recebe texto **gerado pelo modelo**. O programa exige que a resposta cite IDs reais do contexto. Se faltarem citações válidas, o programa bloqueia a resposta; o fato de um ID ser válido não garante, sozinho, que toda interpretação esteja correta. Faça verificação humana na avaliação do projeto.

## Erros comuns

- `Ollama não encontrado`: instale o aplicativo, abra-o e tente `ollama serve`.
- `model not found`: rode `ollama pull gemma3:4b` ou selecione um modelo presente em `ollama list`.
- Lentidão: troque para `gemma3:1b`, reduza o número de evidências para 5.
- `No module named ...`: execute as instalações com o Python do `.venv`, não o Python global.
- O programa respondeu `llm_error`: confira o aviso na tela; o modo não se declara generativo quando o LLM falha.
