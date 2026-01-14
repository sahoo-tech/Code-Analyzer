

import os
from abc import ABC, abstractmethod
from typing import Optional, AsyncIterator
from dataclasses import dataclass

from analyzer.rag.config import LLMConfig
from analyzer.logging_config import get_logger

logger = get_logger("rag.llm")


@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None


class LLMProvider(ABC):
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        pass
    
    @abstractmethod
    def generate(self, prompt: str, context: str = "") -> LLMResponse:
        pass
    
    async def generate_stream(
        self, prompt: str, context: str = ""
    ) -> AsyncIterator[str]:
        response = self.generate(prompt, context)
        yield response.content


class OpenAILLM(LLMProvider):
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
    
    @property
    def model_name(self) -> str:
        return self.config.model
    
    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            self._client = OpenAI(api_key=api_key)
        return self._client
    
    def generate(self, prompt: str, context: str = "") -> LLMResponse:
        client = self._get_client()
        
        messages = []
        if context:
            messages.append({
                "role": "system",
                "content": f"You are a helpful code analysis assistant. Use the following code context to answer questions:\n\n{context}"
            })
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.config.model,
            provider="openai",
            tokens_used=response.usage.total_tokens if response.usage else None,
            finish_reason=response.choices[0].finish_reason,
        )
    
    async def generate_stream(
        self, prompt: str, context: str = ""
    ) -> AsyncIterator[str]:
        client = self._get_client()
        
        messages = []
        if context:
            messages.append({
                "role": "system",
                "content": f"You are a helpful code analysis assistant. Use the following code context to answer questions:\n\n{context}"
            })
        messages.append({"role": "user", "content": prompt})
        
        stream = client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicLLM(LLMProvider):
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
    
    @property
    def model_name(self) -> str:
        return self.config.anthropic_model
    
    def _get_client(self):
        if self._client is None:
            import anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client
    
    def generate(self, prompt: str, context: str = "") -> LLMResponse:
        client = self._get_client()
        
        system_prompt = "You are a helpful code analysis assistant."
        if context:
            system_prompt += f"\n\nUse the following code context to answer questions:\n\n{context}"
        
        response = client.messages.create(
            model=self.config.anthropic_model,
            max_tokens=self.config.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        
        return LLMResponse(
            content=response.content[0].text,
            model=self.config.anthropic_model,
            provider="anthropic",
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            finish_reason=response.stop_reason,
        )
    
    async def generate_stream(
        self, prompt: str, context: str = ""
    ) -> AsyncIterator[str]:
        client = self._get_client()
        
        system_prompt = "You are a helpful code analysis assistant."
        if context:
            system_prompt += f"\n\nUse the following code context to answer questions:\n\n{context}"
        
        with client.messages.stream(
            model=self.config.anthropic_model,
            max_tokens=self.config.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text


class GoogleLLM(LLMProvider):
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
        self._genai = None
    
    @property
    def model_name(self) -> str:
        return self.config.google_model
    
    def _get_client(self):
        if self._client is None:
            import google.generativeai as genai
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not set")
            genai.configure(api_key=api_key)
            self._genai = genai
            self._client = genai.GenerativeModel(
                self.config.google_model,
                system_instruction="You are a helpful code analysis assistant. Analyze code thoroughly and provide detailed, accurate answers."
            )
        return self._client
    
    def generate(self, prompt: str, context: str = "") -> LLMResponse:
        client = self._get_client()
        
        full_prompt = prompt
        if context:
            full_prompt = f"Use the following code context to answer the question:\n\n{context}\n\nQuestion: {prompt}"
        
        response = client.generate_content(
            full_prompt,
            generation_config=self._genai.GenerationConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
            )
        )
        
        return LLMResponse(
            content=response.text,
            model=self.config.google_model,
            provider="google",
            tokens_used=response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else None,
        )
    
    async def generate_stream(
        self, prompt: str, context: str = ""
    ) -> AsyncIterator[str]:
        client = self._get_client()
        
        full_prompt = prompt
        if context:
            full_prompt = f"Use the following code context to answer the question:\n\n{context}\n\nQuestion: {prompt}"
        
        response = client.generate_content(
            full_prompt,
            generation_config=self._genai.GenerationConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
            ),
            stream=True,
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text


class MockLLM(LLMProvider):
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @property
    def model_name(self) -> str:
        return "mock-model"
    
    def generate(self, prompt: str, context: str = "") -> LLMResponse:
        context_summary = ""
        if context:
            lines = context.split("\n")
            entities = []
            for line in lines:
                if line.startswith("[") and "]" in line:
                    entity = line.split("]")[0] + "]"
                    entities.append(entity)
            if entities:
                context_summary = f"\n\nBased on the following code entities:\n" + "\n".join(entities[:5])
        
        response = f"""Based on the analysis of the codebase, here is my response to your query:

**Query:** {prompt}

**Analysis:**
The codebase contains relevant information related to your question.{context_summary}

**Note:** This is a mock response. Set one of these environment variables to use real AI:
- GOOGLE_API_KEY or GEMINI_API_KEY for Google Gemini
- OPENAI_API_KEY for OpenAI GPT
- ANTHROPIC_API_KEY for Anthropic Claude"""

        return LLMResponse(
            content=response,
            model="mock-model",
            provider="mock",
            tokens_used=len(response.split()),
        )
    
    async def generate_stream(
        self, prompt: str, context: str = ""
    ) -> AsyncIterator[str]:
        response = self.generate(prompt, context)
        for word in response.content.split():
            yield word + " "


def detect_available_provider() -> Optional[str]:
    """Detect which AI provider is available based on environment variables."""
    # Priority order: Google (free tier available), OpenAI, Anthropic
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        try:
            import google.generativeai
            return "google"
        except ImportError:
            pass
    
    if os.getenv("OPENAI_API_KEY"):
        try:
            import openai
            return "openai"
        except ImportError:
            pass
    
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            return "anthropic"
        except ImportError:
            pass
    
    return None


def get_llm_provider(config: LLMConfig, auto_detect: bool = True) -> LLMProvider:
    """Get LLM provider based on config.
    
    If auto_detect is True and provider is set to 'auto' or API key is missing,
    automatically selects an available provider.
    """
    provider_type = config.provider.lower()
    
    # Auto-detect mode: try to find any available provider
    if provider_type == "auto" or (auto_detect and provider_type != "mock"):
        detected = detect_available_provider()
        if detected:
            logger.info(f"Auto-detected LLM provider: {detected}")
            provider_type = detected
        elif provider_type == "auto":
            logger.warning("No AI provider detected, using mock. Set GOOGLE_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY")
            return MockLLM(config)
    
    if provider_type == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            if auto_detect:
                detected = detect_available_provider()
                if detected:
                    return get_llm_provider_direct(detected, config)
            logger.warning("OPENAI_API_KEY not set, using mock LLM")
            return MockLLM(config)
        try:
            return OpenAILLM(config)
        except ImportError:
            logger.warning("OpenAI package not installed. Install with: pip install openai")
            return MockLLM(config)
    
    elif provider_type == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            if auto_detect:
                detected = detect_available_provider()
                if detected:
                    return get_llm_provider_direct(detected, config)
            logger.warning("ANTHROPIC_API_KEY not set, using mock LLM")
            return MockLLM(config)
        try:
            return AnthropicLLM(config)
        except ImportError:
            logger.warning("Anthropic package not installed. Install with: pip install anthropic")
            return MockLLM(config)
    
    elif provider_type == "google" or provider_type == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            if auto_detect:
                detected = detect_available_provider()
                if detected:
                    return get_llm_provider_direct(detected, config)
            logger.warning("GOOGLE_API_KEY/GEMINI_API_KEY not set, using mock LLM")
            return MockLLM(config)
        try:
            return GoogleLLM(config)
        except ImportError:
            logger.warning("Google GenAI package not installed. Install with: pip install google-generativeai")
            return MockLLM(config)
    
    elif provider_type == "mock":
        return MockLLM(config)
    
    else:
        raise ValueError(f"Unknown LLM provider: {provider_type}")


def get_llm_provider_direct(provider_type: str, config: LLMConfig) -> LLMProvider:
    """Get a specific LLM provider without fallback."""
    if provider_type == "openai":
        return OpenAILLM(config)
    elif provider_type == "anthropic":
        return AnthropicLLM(config)
    elif provider_type == "google":
        return GoogleLLM(config)
    else:
        return MockLLM(config)
