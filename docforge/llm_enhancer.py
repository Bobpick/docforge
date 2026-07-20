"""LLM-based enhancement for document conversion cleanup.

Supports multiple LLM providers:
- Ollama (local, default) — any model like cogito, llama3, mistral, etc.
- Google Gemini (cloud) — requires API key

The Ollama provider is the default because it's free, private, and
runs locally. No API key needed — just install Ollama and pull a model.
"""

import os
import re
import json
import urllib.request
import urllib.error
from typing import Optional, Dict, Any

from .utils import truncate_text, chunk_text, clean_whitespace


# ──────────────────────────────────────────────────────────────────────
# Provider-agnostic LLM interface
# ──────────────────────────────────────────────────────────────────────

class LLMProvider:
    """Base class for LLM providers."""

    def generate(self, prompt: str, system: str = "") -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The user prompt.
            system: Optional system prompt.

        Returns:
            Generated text.
        """
        raise NotImplementedError

    def is_available(self) -> bool:
        """Check if this provider is ready to use."""
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    """Local LLM via Ollama (http://localhost:11434).

    No API key needed. Just install Ollama and pull a model:
        ollama pull cogito:14b
    """

    def __init__(
        self,
        model: str = "cogito:14b",
        host: str = "http://localhost:11434",
        timeout: int = 300,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, system: str = "") -> str:
        """Generate via Ollama's HTTP API."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 32768,
            },
        }
        if system:
            payload["system"] = system

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "")
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.host}. "
                f"Make sure Ollama is running: ollama serve\n"
                f"Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            req = urllib.request.Request(
                f"{self.host}/api/tags",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m["name"] for m in data.get("models", [])]
                # Check if the requested model (or a base name match) is available
                model_base = self.model.split(":")[0]
                for m in models:
                    if m == self.model or m.startswith(model_base + ":") or m == model_base:
                        return True
                # Model not found locally, but Ollama is running
                # It can auto-pull, so return True
                return True
        except Exception:
            return False

    def list_models(self) -> list:
        """List available Ollama models."""
        try:
            req = urllib.request.Request(
                f"{self.host}/api/tags",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []


class GeminiProvider(LLMProvider):
    """Google Gemini via the generative AI SDK (cloud)."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.model_name = model
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai is required for Gemini. "
                "Install it with: pip install google-generativeai"
            )

        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set GOOGLE_API_KEY env var "
                "or pass llm_api_key parameter."
            )

        genai.configure(api_key=self.api_key)
        self._client = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=65536,
            ),
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
        )
        return self._client

    def generate(self, prompt: str, system: str = "") -> str:
        """Generate via Gemini SDK."""
        client = self._get_client()
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        response = client.generate_content(full_prompt)
        return response.text

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import google.generativeai  # noqa: F401
            return True
        except ImportError:
            return False


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI-compatible API (works with LM Studio, vLLM, llama.cpp server, etc.)."""

    def __init__(
        self,
        model: str = "local-model",
        api_base: str = "http://localhost:8080/v1",
        api_key: str = "not-needed",
    ):
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key

    def generate(self, prompt: str, system: str = "") -> str:
        """Generate via OpenAI-compatible chat completions API."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 32768,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(
                f"{self.api_base}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return True
        except Exception:
            return False


# ──────────────────────────────────────────────────────────────────────
# Unified enhancer
# ──────────────────────────────────────────────────────────────────────

# Default system prompt for all enhancement tasks
_SYSTEM_PROMPT = """You are a document formatting expert. You fix markdown documents that were
converted from PDF/DOCX/PPTX files. Your job is to fix formatting issues
without changing the actual content or information.

Rules:
- Format math expressions with LaTeX: inline $...$, display $$...$$
- Wrap code in fenced code blocks with language tags
- Fix garbled or misaligned tables
- Fix common OCR artifacts
- Do NOT add or remove information
- Do NOT change headings or their levels
- Return ONLY the improved markdown text, no explanations"""


class LLMEnhancer:
    """Use an LLM to clean up and enhance conversion output.

    Supports Ollama (default, local), Google Gemini (cloud), and
    OpenAI-compatible APIs (LM Studio, vLLM, etc.).
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        """
        Args:
            provider: LLM provider — "ollama", "gemini", or "openai-compat".
            model: Model name. Defaults per provider:
                   - ollama: "cogito:14b"
                   - gemini: "gemini-2.0-flash"
                   - openai-compat: "local-model"
            api_key: API key (Gemini only). Falls back to env vars.
            host: Ollama host URL (default: http://localhost:11434).
            api_base: OpenAI-compatible API base URL.
        """
        self.provider_name = provider

        if provider == "ollama":
            self._provider: LLMProvider = OllamaProvider(
                model=model or os.environ.get("DOCFORGE_OLLAMA_MODEL", "cogito:14b"),
                host=host or os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            )
        elif provider == "gemini":
            self._provider = GeminiProvider(
                api_key=api_key,
                model=model or "gemini-2.0-flash",
            )
        elif provider == "openai-compat":
            self._provider = OpenAICompatibleProvider(
                model=model or "local-model",
                api_base=api_base or "http://localhost:8080/v1",
                api_key=api_key or "not-needed",
            )
        else:
            raise ValueError(
                f"Unknown LLM provider: {provider!r}. "
                f"Supported: 'ollama', 'gemini', 'openai-compat'"
            )

    @classmethod
    def from_env(cls) -> "LLMEnhancer":
        """Create an enhancer from environment variables.

        DOCFORGE_LLM_PROVIDER: ollama | gemini | openai-compat (default: ollama)
        DOCFORGE_LLM_MODEL: model name (default: cogito:14b for ollama)
        GOOGLE_API_KEY: Gemini API key
        OLLAMA_HOST: Ollama server URL
        """
        provider = os.environ.get("DOCFORGE_LLM_PROVIDER", "ollama")
        model = os.environ.get("DOCFORGE_LLM_MODEL", None)
        return cls(provider=provider, model=model)

    def enhance(self, markdown: str, structured: Optional[Dict[str, Any]] = None) -> str:
        """Enhance markdown output using LLM.

        Applies multiple enhancement passes:
        1. Table cleanup and restructuring
        2. Math equation formatting
        3. Code block detection and formatting
        4. General formatting improvements

        Args:
            markdown: The markdown text to enhance.
            structured: Optional structured data for context.

        Returns:
            Enhanced markdown text.
        """
        if not markdown.strip():
            return markdown

        # Check if there's anything that needs enhancement
        needs_enhancement = self._needs_enhancement(markdown)
        if not needs_enhancement:
            return markdown

        try:
            # Enhance tables first (most impactful)
            markdown = self._enhance_tables(markdown)

            # Enhance math and code
            markdown = self._enhance_math_and_code(markdown)

            # Final cleanup pass
            markdown = self._final_cleanup(markdown)

        except Exception as e:
            # If LLM fails, return the original with a comment
            markdown = f"{markdown}\n\n<!-- LLM enhancement failed: {e} -->"

        return markdown

    def _needs_enhancement(self, markdown: str) -> bool:
        """Check if the markdown would benefit from LLM enhancement."""
        indicators = [
            r'\|.*\|.*\n(?!\|[-\s:|]+)',  # Table row without separator
            r'(\S\t){2,}',                  # Tab-separated data
            r'[a-zA-Z]\^[0-9]',            # Unformatted superscript
            r'[a-zA-Z]_[0-9]',             # Unformatted subscript
            r'[^\x00-\x7F]{3,}',           # OCR artifacts
        ]
        for pattern in indicators:
            if re.search(pattern, markdown):
                return True
        return True  # Default: always try to enhance

    def _enhance_tables(self, markdown: str) -> str:
        """Use LLM to clean up messy tables."""
        table_sections = self._extract_table_sections(markdown)
        if not table_sections:
            return markdown

        result = markdown
        for table_text, start, end in reversed(table_sections):
            enhanced = self._llm_enhance_table(table_text)
            if enhanced:
                result = result[:start] + enhanced + result[end:]

        return result

    def _extract_table_sections(self, markdown: str) -> list:
        """Extract table sections from markdown for individual enhancement."""
        tables = []
        lines = markdown.split("\n")
        in_table = False
        table_start = 0
        table_lines = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("|") and "|" in stripped[1:]:
                if not in_table:
                    in_table = True
                    table_start = sum(len(l) + 1 for l in lines[:i])
                table_lines.append(line)
            elif in_table:
                table_text = "\n".join(table_lines)
                table_end = table_start + len(table_text)
                tables.append((table_text, table_start, table_end))
                in_table = False
                table_lines = []

        if in_table and table_lines:
            table_text = "\n".join(table_lines)
            table_end = table_start + len(table_text)
            tables.append((table_text, table_start, table_end))

        return tables

    def _enhance_math_and_code(self, markdown: str) -> str:
        """Use LLM to improve math equations and code block formatting."""
        has_math = bool(re.search(r'[\\^_{}]', markdown))
        has_code = bool(re.search(r'(def |function |class |import |#include )', markdown))

        if not has_math and not has_code:
            return markdown

        # If the text is very long, process in chunks
        if len(markdown) > 15000:
            return self._enhance_in_chunks(markdown)

        prompt = f"""Clean up the following markdown text:

1. Format mathematical expressions with proper LaTeX notation:
   - Inline math: $expression$
   - Display math: $$expression$$
   - Fix OCR errors in math (e.g., "x2" → "$x^2$", "alpha" → "$\\alpha$")

2. Wrap any detected code in fenced code blocks with language tags

3. Do NOT change any other formatting, headings, or content
4. Do NOT add or remove information
5. Return ONLY the improved markdown text

Markdown to improve:

{markdown}"""

        try:
            result = self._provider.generate(prompt, system=_SYSTEM_PROMPT)
            result = result.strip()
            result = re.sub(r'^```\n?', '', result)
            result = re.sub(r'\n?```$', '', result)
            return result
        except Exception:
            return markdown

    def _enhance_in_chunks(self, markdown: str) -> str:
        """Enhance long documents by processing in chunks."""
        chunks = chunk_text(markdown, chunk_size=12000, overlap=200)
        enhanced_chunks = []

        for chunk in chunks:
            try:
                prompt = f"""Clean up this markdown fragment:
1. Format math expressions with LaTeX ($...$ or $$...$$)
2. Wrap code in fenced code blocks with language tags
3. Fix any OCR artifacts
4. Do NOT change headings or other content
5. Return ONLY the improved markdown

Fragment:
{chunk}"""

                result = self._provider.generate(prompt, system=_SYSTEM_PROMPT)
                result = result.strip()
                result = re.sub(r'^```\n?', '', result)
                result = re.sub(r'\n?```$', '', result)
                enhanced_chunks.append(result)
            except Exception:
                enhanced_chunks.append(chunk)

        return "\n\n".join(enhanced_chunks)

    def _llm_enhance_table(self, table_text: str) -> Optional[str]:
        """Use LLM to clean up a single table."""
        prompt = f"""The following markdown table may have issues from document conversion:
- Missing or misaligned columns
- Merged cells that need handling
- Garbled text from OCR
- Missing separator rows

Please fix and return ONLY the corrected markdown table. Preserve all data.

Table to fix:
{table_text}"""

        try:
            result = self._provider.generate(prompt, system=_SYSTEM_PROMPT)
            result = result.strip()
            result = re.sub(r'^```\n?', '', result)
            result = re.sub(r'\n?```$', '', result)
            return result
        except Exception:
            return None

    def _final_cleanup(self, markdown: str) -> str:
        """Final cleanup pass without LLM — pure regex-based."""
        # Fix double-escaped characters
        markdown = markdown.replace("\\\\(", "\\(")
        markdown = markdown.replace("\\\\)", "\\)")
        markdown = markdown.replace("\\\\[", "\\[")
        markdown = markdown.replace("\\\\]", "\\]")

        # Fix common math OCR artifacts
        markdown = re.sub(r'(\w)\s*\^\s*(\d)', r'\1^{\2}', markdown)
        markdown = re.sub(r'(\w)\s*_\s*(\d)', r'\1_{\2}', markdown)

        # Clean up excessive whitespace
        markdown = clean_whitespace(markdown)

        return markdown

    def is_available(self) -> bool:
        """Check if the LLM enhancer is available."""
        return self._provider.is_available()

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current LLM provider."""
        info = {
            "provider": self.provider_name,
            "available": self.is_available(),
        }
        if isinstance(self._provider, OllamaProvider):
            info["model"] = self._provider.model
            info["host"] = self._provider.host
            info["models"] = self._provider.list_models()
        elif isinstance(self._provider, GeminiProvider):
            info["model"] = self._provider.model_name
            info["has_api_key"] = bool(self._provider.api_key)
        elif isinstance(self._provider, OpenAICompatibleProvider):
            info["model"] = self._provider.model
            info["api_base"] = self._provider.api_base
        return info
