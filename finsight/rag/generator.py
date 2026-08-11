"""Grounded answer generation for FinSight RAG."""

from __future__ import annotations

import os
import re
from typing import Any

from dotenv import load_dotenv


NOT_FOUND_ANSWER = "I could not find this information in the uploaded document."


class RAGGenerator:
    """Generate answers from retrieved chunks using a grounded prompt."""

    def __init__(
        self,
        llm_provider: str | None = None,
        anthropic_model: str | None = None,
        openai_model: str | None = None,
        anthropic_client: Any | None = None,
        openai_client: Any | None = None,
    ):
        load_dotenv()
        self.llm_provider = (llm_provider or os.getenv("FINSIGHT_LLM_PROVIDER") or "anthropic").strip().lower()
        self.anthropic_model = anthropic_model or os.getenv("ANTHROPIC_MODEL") or "claude-3-5-sonnet-20241022"
        self.openai_model = openai_model or os.getenv("OPENAI_MODEL") or "gpt-5"
        self._anthropic_client = anthropic_client
        self._openai_client = openai_client
        self._last_tokens_used = 0
        self.system_prompt = (
            "You are FinSight's document-grounded RAG analyst. Answer only from the provided excerpts. "
            f"If the excerpts do not contain the answer, say exactly: \"{NOT_FOUND_ANSWER}\" "
            "Every factual claim from the document must include a [Page X] citation. Do not use outside knowledge."
        )

        if self.llm_provider not in {"anthropic", "openai"}:
            raise ValueError("llm_provider must be either 'anthropic' or 'openai'.")

    def generate(self, query: str, retrieved_chunks: list[dict], max_tokens: int = 1024) -> dict:
        """Generate a grounded answer from retrieved chunks."""
        if not retrieved_chunks:
            return {
                "answer": NOT_FOUND_ANSWER,
                "citations": [],
                "chunks_used": 0,
                "error": None,
            }

        prompt = self._build_prompt(query, retrieved_chunks)
        error = None
        self._last_tokens_used = 0
        try:
            if not self._provider_configured():
                answer = self._extractive_fallback(retrieved_chunks)
                error = f"{self.llm_provider} API key not configured; returned extractive fallback."
            elif self.llm_provider == "openai":
                answer = self._generate_openai(prompt, max_tokens=max_tokens)
            else:
                answer = self._generate_anthropic(prompt, max_tokens=max_tokens)
        except Exception as exc:
            answer = self._extractive_fallback(retrieved_chunks)
            error = str(exc)

        answer = self._ensure_grounded_answer(answer, retrieved_chunks)
        return {
            "answer": answer,
            "citations": self._extract_citations(answer),
            "chunks_used": len(retrieved_chunks),
            "model_used": self.anthropic_model if self.llm_provider == "anthropic" else self.openai_model,
            "tokens_used": self._last_tokens_used,
            "error": error,
        }

    def _build_prompt(self, query: str, retrieved_chunks: list[dict]) -> str:
        excerpts = []
        for index, chunk in enumerate(retrieved_chunks, start=1):
            metadata = chunk.get("metadata", {})
            page_start = metadata.get("page_start", "?")
            page_end = metadata.get("page_end", page_start)
            if page_start == page_end:
                page_label = f"Page {page_start}"
            else:
                page_label = f"Pages {page_start}-{page_end}"
            excerpts.append(f"[Excerpt {index} | {page_label}]\n{chunk.get('text', '')}")

        return (
            f"Question:\n{query}\n\n"
            "Retrieved excerpts:\n"
            + "\n\n".join(excerpts)
            + "\n\nAnswer with concise financial analysis and page citations."
        )

    def _generate_anthropic(self, prompt: str, max_tokens: int) -> str:
        client = self._anthropic_client
        if client is None:
            from anthropic import Anthropic

            client = Anthropic()
            self._anthropic_client = client

        response = client.messages.create(
            model=self.anthropic_model,
            max_tokens=max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._last_tokens_used = int(getattr(usage, "input_tokens", 0) or 0) + int(
                getattr(usage, "output_tokens", 0) or 0
            )
        return "".join(block.text for block in response.content if getattr(block, "type", None) == "text")

    def _generate_openai(self, prompt: str, max_tokens: int) -> str:
        client = self._openai_client
        if client is None:
            from openai import OpenAI

            client = OpenAI()
            self._openai_client = client

        response = client.responses.create(
            model=self.openai_model,
            instructions=self.system_prompt,
            input=prompt,
            max_output_tokens=max_tokens,
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            total = getattr(usage, "total_tokens", None)
            self._last_tokens_used = int(total) if total is not None else int(
                getattr(usage, "input_tokens", 0) or 0
            ) + int(getattr(usage, "output_tokens", 0) or 0)
        output_text = getattr(response, "output_text", None)
        if output_text:
            return str(output_text)

        parts: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    parts.append(str(text))
        return "".join(parts)

    def _provider_configured(self) -> bool:
        if self.llm_provider == "anthropic":
            return self._anthropic_client is not None or bool(os.getenv("ANTHROPIC_API_KEY"))
        return self._openai_client is not None or bool(os.getenv("OPENAI_API_KEY"))

    def _extractive_fallback(self, retrieved_chunks: list[dict]) -> str:
        lines = ["Based on the uploaded document:"]
        for chunk in retrieved_chunks[:2]:
            metadata = chunk.get("metadata", {})
            page = metadata.get("page_start", "?")
            text = str(chunk.get("text", "")).strip()
            lines.append(f"- {text} [Page {page}]")
        return "\n".join(lines)

    def _ensure_grounded_answer(self, answer: str, retrieved_chunks: list[dict]) -> str:
        answer = (answer or "").strip()
        if not answer:
            return NOT_FOUND_ANSWER
        if answer == NOT_FOUND_ANSWER or "[Page " in answer:
            return answer

        first_page = retrieved_chunks[0].get("metadata", {}).get("page_start")
        if first_page is None:
            return answer
        return f"{answer} [Page {first_page}]"

    def _extract_citations(self, answer: str) -> list[int]:
        pages = {int(match) for match in re.findall(r"\[Page (\d+)\]", answer)}
        return sorted(pages)
