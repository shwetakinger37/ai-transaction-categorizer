from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

from django.conf import settings

logger = logging.getLogger(__name__)

_MAX_AGENTIC_ITERATIONS = 10




class BaseLLMClient(ABC):
    

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Single-turn text completion — no tool use."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable identifier for logging."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier (openai / huggingface / anthropic)."""

    def supports_tools(self) -> bool:
        """Return True if this client implements complete_agentic()."""
        return False

    def complete_agentic(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: List[Dict[str, Any]],
        tool_executor: Callable[[str, Dict[str, Any]], str],
    ) -> str:
        """
        Agentic multi-turn completion with tool use.
        Default implementation ignores tools and falls back to complete().
        Providers that support tool use must override this method.
        """
        return self.complete(system_prompt, user_prompt)

    # Populated by complete_agentic() implementations; read by the service layer.
    last_tools_used: List[str] = []


class OpenAIClient(BaseLLMClient):
    def __init__(self) -> None:
        from openai import OpenAI  # lazy import — not a hard dep if unused

        kwargs: Dict[str, Any] = {"api_key": settings.LLM_API_KEY}
        if settings.LLM_BASE_URL:
            kwargs["base_url"] = settings.LLM_BASE_URL   # supports HuggingFace TGI, vLLM, etc.

        self._client = OpenAI(**kwargs)
        self._model = settings.LLM_MODEL

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "openai"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        logger.debug("LLM request | provider=openai model=%s", self._model)
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = response.choices[0].message.content or ""
        logger.debug("LLM response received | length=%d", len(text))
        return text



class HuggingFaceClient(OpenAIClient):


    @property
    def provider_name(self) -> str:
        return "huggingface"




class AnthropicClient(BaseLLMClient):
    def __init__(self) -> None:
        import anthropic  # lazy import

        self._client = anthropic.Anthropic(api_key=settings.LLM_API_KEY)
        self._model = settings.LLM_MODEL or "claude-haiku-4-5-20251001"
        self.last_tools_used: List[str] = []

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def supports_tools(self) -> bool:
        return True

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Single-turn completion — no tools. Used as fallback."""
        logger.debug("LLM request | provider=anthropic model=%s", self._model)
        message = self._client.messages.create(
            model=self._model,
            max_tokens=settings.LLM_MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = message.content[0].text if message.content else ""
        logger.debug("LLM response received | length=%d", len(text))
        return text

    def complete_agentic(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: List[Dict[str, Any]],
        tool_executor: Callable[[str, Dict[str, Any]], str],
    ) -> str:
        
        self.last_tools_used = []
        messages: List[Dict[str, Any]] = [{"role": "user", "content": user_prompt}]

        logger.debug(
            "Agentic request | provider=anthropic model=%s tools=%d",
            self._model,
            len(tools),
        )

        for iteration in range(_MAX_AGENTIC_ITERATIONS):
            response = self._client.messages.create(
                model=self._model,
                max_tokens=settings.LLM_MAX_TOKENS,
                system=system_prompt,
                tools=tools,
                messages=messages,
            )

            logger.debug(
                "Agentic iteration %d | stop_reason=%s",
                iteration + 1,
                response.stop_reason,
            )

            if response.stop_reason == "end_turn":
                text = next(
                    (b.text for b in response.content if hasattr(b, "text")), ""
                )
                logger.debug(
                    "Agentic loop complete | iterations=%d tools_used=%s",
                    iteration + 1,
                    self.last_tools_used,
                )
                return text

            if response.stop_reason == "tool_use":
                # Append assistant's turn (including tool_use blocks)
                messages.append({"role": "assistant", "content": response.content})

                # Execute every tool the model requested
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        self.last_tools_used.append(block.name)
                        result = tool_executor(block.name, block.input)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result,
                            }
                        )

                # Feed results back as a user turn
                messages.append({"role": "user", "content": tool_results})

            else:
                # Unexpected stop reason (e.g. max_tokens) — return what we have
                logger.warning(
                    "Unexpected stop_reason '%s' at iteration %d",
                    response.stop_reason,
                    iteration + 1,
                )
                return next(
                    (b.text for b in response.content if hasattr(b, "text")), ""
                )

        logger.warning(
            "Agentic loop hit max iterations (%d); returning last text content",
            _MAX_AGENTIC_ITERATIONS,
        )
        return next((b.text for b in response.content if hasattr(b, "text")), "")


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────

class LLMClientFactory:
    _REGISTRY: Dict[str, type] = {
        "openai": OpenAIClient,
        "huggingface": HuggingFaceClient,
        "anthropic": AnthropicClient,
    }

    @classmethod
    def get(cls) -> BaseLLMClient:
        provider = (settings.LLM_PROVIDER or "openai").lower()
        client_cls = cls._REGISTRY.get(provider)
        if not client_cls:
            raise ValueError(
                f"Unknown LLM_PROVIDER '{provider}'. "
                f"Valid options: {list(cls._REGISTRY.keys())}"
            )
        return client_cls()



def extract_json_block(raw: str) -> Dict[str, Any]:
    
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    
    brace = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace:
        return json.loads(brace.group(0))

    raise ValueError(f"No JSON object found in LLM response: {raw[:200]!r}")
