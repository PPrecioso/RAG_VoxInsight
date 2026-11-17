"""Conectores de geração: Ollama local ou OpenAI API. Sem chave não implica sem LLM."""
from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import Protocol

import requests


class LLMError(RuntimeError):
    """Erro de configuração, indisponibilidade ou geração do modelo."""


class Generator(Protocol):
    def generate(self, system: str, user: str) -> str: ...


@dataclass
class OllamaGenerator:
    model: str = "gemma3:4b"
    base_url: str = "http://127.0.0.1:11434"
    timeout: int = 300

    def generate(self, system: str, user: str) -> str:
        url = self.base_url.rstrip("/") + "/api/chat"
        try:
            resp = requests.post(url, json={"model": self.model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "stream": False, "options": {"temperature": 0, "num_predict": 1100}},
                timeout=self.timeout)
            resp.raise_for_status()
            payload = resp.json()
            content = payload.get("message", {}).get("content", "")
            if not isinstance(content, str) or not content.strip():
                raise LLMError("Ollama retornou uma resposta vazia.")
            return content.strip()
        except requests.exceptions.ConnectionError as exc:
            raise LLMError("O Ollama não está em execução. Instale/inicie Ollama e rode 'ollama pull gemma3:4b'.") from exc
        except requests.exceptions.Timeout as exc:
            raise LLMError("O modelo local excedeu o tempo limite. Tente um modelo menor, como gemma3:1b.") from exc
        except requests.exceptions.HTTPError as exc:
            detail = (exc.response.text or "")[:350] if exc.response is not None else ""
            raise LLMError(f"Erro HTTP do Ollama: {detail}. Confirme 'ollama pull {self.model}'.") from exc
        except requests.exceptions.RequestException as exc:
            raise LLMError(f"Não foi possível consultar o Ollama: {exc}") from exc


@dataclass
class OpenAIGenerator:
    model: str = "gpt-4.1-mini"
    api_key: str | None = None

    def generate(self, system: str, user: str) -> str:
        key = self.api_key or os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise LLMError("Chave OPENAI_API_KEY não configurada. Use Ollama local ou configure a chave no .env.")
        try:
            from openai import OpenAI
            client = OpenAI(api_key=key, timeout=120, max_retries=1)
            response = client.chat.completions.create(
                model=self.model, temperature=0,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            answer = response.choices[0].message.content or ""
            if not answer.strip():
                raise LLMError("O modelo respondeu sem conteúdo.")
            return answer.strip()
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Falha na API de geração ({type(exc).__name__}): {str(exc)[:250]}") from exc


def select_generator(provider: str, model: str = "", api_key: str | None = None,
                     base_url: str | None = None) -> Generator | None:
    provider = provider.lower().strip()
    if provider == "ollama":
        return OllamaGenerator(model=model or os.getenv("OLLAMA_MODEL", "gemma3:4b"),
                               base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    if provider == "openai":
        return OpenAIGenerator(model=model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), api_key=api_key)
    if provider == "extrativo":
        return None
    raise ValueError("Provedor inválido. Selecione ollama, openai ou extrativo.")


def check_ollama(base_url: str | None = None) -> list[str]:
    """Lista os modelos locais disponíveis, sem iniciar downloads automaticamente."""
    url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/") + "/api/tags"
    try:
        response = requests.get(url, timeout=3)
        response.raise_for_status()
        return [m.get("name", "") for m in response.json().get("models", []) if m.get("name")]
    except (requests.RequestException, ValueError):
        return []
