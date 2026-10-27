"""LLM abstractions for text generation and streaming."""

from abc import ABC, abstractmethod
import os
import re
from typing import Iterator, Optional


class BaseLLM(ABC):
    """Abstract interface for Language Model completions."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate a complete response."""
        pass

    @abstractmethod
    def stream(self, prompt: str, system_prompt: str = "") -> Iterator[str]:
        """Stream chunks of the generated response."""
        pass


class OpenAILLM(BaseLLM):
    """OpenAI Chat Completion LLM."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.2,
        api_key: Optional[str] = None,
    ):
        self._model_name = model_name
        self.temperature = temperature

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")

    @property
    def model_name(self) -> str:
        return self._model_name

    def _build_messages(self, prompt: str, system_prompt: str = "") -> list[dict]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        messages = self._build_messages(prompt, system_prompt)
        response = self.client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    def stream(self, prompt: str, system_prompt: str = "") -> Iterator[str]:
        messages = self._build_messages(prompt, system_prompt)
        response = self.client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=self.temperature,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content if chunk.choices else ""
            if delta:
                yield delta


class LocalMockLLM(BaseLLM):
    """Offline, deterministic LLM fallback that synthesizes answers directly from context.

    Useful for offline testing and environments without API credentials.
    """

    def __init__(self, model_name: str = "local-heuristic-synthesizer"):
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        # Extract context and question from prompt format
        if "Context:" in prompt and "Question:" in prompt:
            context_part = prompt.split("Context:")[1].split("Question:")[0].strip()
            question_part = prompt.split("Question:")[1].strip()
            
            # Simple keyword matching to find most relevant sentences in context
            keywords = [w.lower() for w in re.findall(r"\b\w{3,}\b", question_part)]
            sentences = [s.strip() for s in re.split(r"[.!\n]+", context_part) if s.strip()]

            scored_sentences = []
            for s in sentences:
                score = sum(1 for kw in keywords if kw in s.lower())
                if score > 0:
                    scored_sentences.append((score, s))

            scored_sentences.sort(key=lambda x: x[0], reverse=True)

            if scored_sentences:
                top_facts = " ".join(s for _, s in scored_sentences[:3])
                return f"[Local Offline Mode] Based on the indexed documentation:\n\n{top_facts}."
            else:
                return f"[Local Offline Mode] Found retrieved context, but no specific sentence directly matched keywords in '{question_part}'."

        return f"[Local Offline Mode] Received prompt: {prompt[:100]}..."

    def stream(self, prompt: str, system_prompt: str = "") -> Iterator[str]:
        full_text = self.generate(prompt, system_prompt)
        words = full_text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")


def is_valid_openai_key(key: Optional[str]) -> bool:
    if not key:
        return False
    key = key.strip()
    return (key.startswith("sk-") or key.startswith("sess-")) and len(key) >= 20


def get_llm(provider: str = "auto", model_name: str = "gpt-4o-mini") -> BaseLLM:
    """Factory to retrieve LLM instance."""
    provider = provider.lower()
    raw_key = os.environ.get("OPENAI_API_KEY", "")
    has_valid_key = is_valid_openai_key(raw_key)

    if provider == "openai":
        return OpenAILLM(model_name=model_name)
    elif provider == "local":
        return LocalMockLLM()
    elif provider == "auto":
        if has_valid_key:
            return OpenAILLM(model_name=model_name)
        else:
            return LocalMockLLM()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Options: 'auto', 'openai', 'local'")
