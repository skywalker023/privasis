#!/usr/bin/env python3
"""
Unified model client for multiple providers.

Supports:
- OpenAI API (GPT-3.5, GPT-4, GPT-5, O1, O3)
- NVIDIA NIM API
- vLLM Server (OpenAI-compatible API)

Model naming conventions:
- "gpt-4.1", "o1-preview" -> OpenAI API
- "nvdev/meta/llama-3.1-8b-instruct" -> NVIDIA NIM API
- Any model name + vllm_server_url -> vLLM Server
"""

import os
import logging
import asyncio
from typing import List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from enum import Enum

# Set up logging
logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Supported model providers."""
    OPENAI = "openai"
    NVIDIA = "nvidia"
    VLLM_SERVER = "vllm-server"  # vLLM running as OpenAI-compatible server


@dataclass
class ProviderConfig:
    """Configuration for an API provider."""
    provider: ModelProvider
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    model_name: Optional[str] = None  # The actual model name to use in API calls


# Provider detection patterns and configurations
PROVIDER_CONFIGS = {
    # OpenAI models
    "gpt-4": ProviderConfig(ModelProvider.OPENAI, api_key_env="OPENAI_API_KEY"),
    "gpt-5": ProviderConfig(ModelProvider.OPENAI, api_key_env="OPENAI_API_KEY"),
    "o1-": ProviderConfig(ModelProvider.OPENAI, api_key_env="OPENAI_API_KEY"),
    "o3": ProviderConfig(ModelProvider.OPENAI, api_key_env="OPENAI_API_KEY"),
    
    # NVIDIA NIM models
    "nvdev/": ProviderConfig(
        ModelProvider.NVIDIA,
        base_url="https://integrate.api.nvidia.com/v1",
        api_key_env="NVDEV_API_KEY"
    ),
}


def detect_provider(model_path: str) -> Tuple[ProviderConfig, str]:
    """Detect the provider and return config with the actual model name.
    
    Args:
        model_path: Model name. Use prefixes to force specific providers:
            - "nvdev/" - Force NVIDIA NIM API
        
    Returns:
        Tuple of (ProviderConfig, actual_model_name)
    """
    # Check against known patterns
    for pattern, config in PROVIDER_CONFIGS.items():
        if pattern.lower() in model_path.lower():
            # For NVIDIA, strip the nvdev/ prefix for the actual API call
            actual_model = model_path
            if pattern == "nvdev/":
                # nvdev/openai/gpt-oss-120b -> openai/gpt-oss-120b
                actual_model = model_path.replace("nvdev/", "", 1)
            return ProviderConfig(
                provider=config.provider,
                base_url=config.base_url,
                api_key_env=config.api_key_env,
                model_name=actual_model
            ), model_path
    
    # Default to OpenAI for unknown models
    return ProviderConfig(ModelProvider.OPENAI, api_key_env="OPENAI_API_KEY"), model_path


def add_provider_config(pattern: str, config: ProviderConfig):
    """Add a custom provider configuration.
    
    Args:
        pattern: Pattern to match against model names (case-insensitive)
        config: ProviderConfig for this pattern
    """
    PROVIDER_CONFIGS[pattern] = config


class UnifiedModelClient:
    """Unified client for multiple model providers.
    
    Automatically detects the provider based on model name and uses the
    appropriate client for generation.
    
    Example:
        >>> # OpenAI API
        >>> client = UnifiedModelClient("gpt-4.1")
        >>> responses = client.generate(["Hello, world!"])
        
        >>> # vLLM server
        >>> client = UnifiedModelClient("meta-llama/Llama-3.1-8B-Instruct", vllm_server_url="http://localhost:8000/v1")
        >>> responses = client.generate(["Hello, world!"])
    """
    
    def __init__(
        self, 
        model_path: str, 
        postprocess_fn: Optional[Callable[[str], str]] = None,
        vllm_server_url: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the model client.
        
        Args:
            model_path: Model name (API) or model name for vLLM server
            postprocess_fn: Optional custom postprocessing function
            vllm_server_url: URL of vLLM server (e.g., http://localhost:8000/v1)
                            If provided, connects to vLLM server for local model inference
            **kwargs: Additional arguments (ignored, for backward compatibility with
                     vLLM local mode parameters like num_gpus, gpu_memory_utilization, etc.)
        """
        # Log ignored kwargs for debugging
        if kwargs:
            logger.debug(f"Ignoring unused parameters: {list(kwargs.keys())}")
        self.model_path = model_path
        self._postprocess_fn = postprocess_fn
        self.vllm_server_url = vllm_server_url
        
        # If vllm_server_url is provided, use vLLM server mode
        if vllm_server_url:
            self.provider = ModelProvider.VLLM_SERVER
            self.actual_model_name = model_path
            self.config = ProviderConfig(ModelProvider.VLLM_SERVER, base_url=vllm_server_url)
            logger.info(f"Using vLLM server at {vllm_server_url} for model: {self.actual_model_name}")
            self._init_vllm_server(vllm_server_url)
            return
        
        # Detect provider and get config
        self.config, self.actual_model_name = detect_provider(model_path)
        self.provider = self.config.provider
        
        logger.info(f"Detected provider: {self.provider.value} for model: {model_path}")
        
        # Initialize based on provider
        if self.provider == ModelProvider.OPENAI:
            self._init_openai()
        elif self.provider == ModelProvider.NVIDIA:
            self._init_openai_compatible()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _init_openai(self):
        """Initialize OpenAI client with proper timeout and retry settings."""
        from openai import OpenAI, AsyncOpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        
        # Configure timeouts and retries for reliability
        # Default timeout is too short for some operations
        timeout_config = 120.0  # 2 minutes
        
        self.client = OpenAI(
            api_key=api_key,
            timeout=timeout_config,
            max_retries=3,
        )
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            timeout=timeout_config,
            max_retries=3,
        )
        self.tokenizer = None
    
    def _init_openai_compatible(self):
        """Initialize OpenAI-compatible client (NVIDIA NIM, etc.)."""
        from openai import OpenAI, AsyncOpenAI
        
        api_key = os.getenv(self.config.api_key_env)
        base_url = self.config.base_url
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.tokenizer = None
    
    def _init_vllm_server(self, server_url: str):
        """Initialize client for vLLM server (OpenAI-compatible API).
        
        Args:
            server_url: URL of the vLLM server (e.g., http://localhost:8000/v1)
        """
        from openai import OpenAI, AsyncOpenAI
        
        # Configure longer timeouts for vLLM server
        # vLLM can take longer to process requests, especially with large models
        timeout_config = 600.0  # 10 minutes - vLLM can take a while for long generations
        
        # vLLM server doesn't require an API key, but OpenAI client needs something
        self.client = OpenAI(
            api_key="EMPTY", 
            base_url=server_url,
            timeout=timeout_config,
            max_retries=2  # Built-in retries for transient errors
        )
        self.async_client = AsyncOpenAI(
            api_key="EMPTY", 
            base_url=server_url,
            timeout=timeout_config,
            max_retries=2  # Built-in retries for transient errors
        )
        self.tokenizer = None
    
    def close(self):
        """Close the client and release resources.
        
        This should be called when done with the client to ensure proper cleanup
        of HTTP connection pools and background threads.
        """
        if hasattr(self, 'client') and self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
        if hasattr(self, 'async_client') and self.async_client is not None:
            try:
                self.async_client.close()
            except Exception:
                pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
        return False

    def _default_postprocess(self, text: str) -> str:
        """Default post-processing for model output."""
        if text is None:
            return ""
        text = text.strip()
        
        # Handle thinking tokens (for models like Qwen3, DeepSeek-R1)
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        
        # Handle gpt-oss-120b style tokens (analysis/final/assistant without angle brackets)
        # These appear when special tokens are decoded without proper handling
        import re
        
        # Pattern: look for "final" marker and extract content after it
        # Handles both "<final>content</final>" and "finalcontent" (stripped brackets)
        if "</final>" in text:
            # Standard XML-style tags
            match = re.search(r'<final>(.*?)</final>', text, re.DOTALL)
            if match:
                text = match.group(1).strip()
        elif "final" in text.lower():
            # Handle stripped special tokens: "analysisWe need...assistantfinalActual answer"
            # Split on common patterns where "final" appears
            patterns = [
                r'(?:assistant)?final\s*',  # "assistantfinal" or "final"
                r'</assistant>.*?final\s*',
            ]
            for pattern in patterns:
                parts = re.split(pattern, text, flags=re.IGNORECASE)
                if len(parts) > 1:
                    text = parts[-1].strip()
                    break
        
        # Also handle "</analysis>" style tags
        if "</analysis>" in text:
            text = text.split("</analysis>")[-1].strip()
        
        # Clean up any remaining assistant markers
        text = re.sub(r'^(assistant|</assistant>)\s*', '', text, flags=re.IGNORECASE)
        
        return text
    
    def _postprocess_output(self, text: str) -> str:
        """Post-process model output."""
        text = self._default_postprocess(text)
        if self._postprocess_fn:
            text = self._postprocess_fn(text)
        return text

    def generate(
        self, 
        prompts: List[str], 
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> List[str]:
        """Generate responses for a batch of prompts.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 for deterministic)
            
        Returns:
            List of generated responses
        """
        if self.provider == ModelProvider.VLLM_SERVER:
            return self._generate_vllm_server(prompts, max_tokens, temperature)
        elif self.provider == ModelProvider.OPENAI:
            return self._generate_openai(prompts, max_tokens, temperature)
        elif self.provider == ModelProvider.NVIDIA:
            return self._generate_openai_compatible(prompts, max_tokens, temperature)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _get_max_concurrent_requests(self) -> int:
        """Get the maximum number of concurrent requests based on provider.
        
        Cloud APIs can handle high concurrency, local models should be limited.
        """
        # Cloud API providers - high concurrency
        cloud_providers = {
            ModelProvider.OPENAI,
            ModelProvider.NVIDIA,
        }
        
        if self.provider in cloud_providers:
            return 128  # Cloud APIs can handle many concurrent requests
        else:
            # vLLM server - limited concurrency
            return 12  # Conservative for local servers
    
    def _is_reasoning_model(self) -> bool:
        """Check if this is a reasoning model (GPT-5, O1, O3, etc.).
        
        Reasoning models have different API requirements:
        - Require max_completion_tokens instead of max_tokens
        - Don't support custom temperature (only default of 1.0)
        - Need higher token limits (reasoning tokens are separate from output)
        """
        model_lower = self.model_path.lower()
        # Reasoning models with new-style API requirements
        reasoning_patterns = ['gpt-5', 'o1-', 'o1', 'o3-', 'o3']
        return any(pattern in model_lower for pattern in reasoning_patterns)
    
    def _get_min_tokens_for_reasoning(self, requested_tokens: int) -> int:
        """Get minimum token limit for reasoning models.
        
        Reasoning models use tokens for internal reasoning before producing output.
        We need to ensure enough tokens for both reasoning AND output.
        """
        # Reasoning models need significantly more tokens
        # Minimum 4096, or 4x the requested amount, whichever is higher
        min_reasoning_tokens = max(4096, requested_tokens * 4)
        return max(requested_tokens, min_reasoning_tokens)
    
    def _generate_openai(
        self, 
        prompts: List[str], 
        max_tokens: int, 
        temperature: float,
        json_mode: bool = False,
        response_format: Optional[Any] = None,
    ) -> List[Any]:
        """Generate using OpenAI API with async batching and concurrency limits.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            json_mode: If True, request JSON object response
            response_format: Pydantic model for structured output (OpenAI only)
        
        Returns:
            List of responses (strings or parsed Pydantic objects)
        """
        from openai import APIConnectionError, APITimeoutError, APIStatusError
        
        # Check if we need structured output (Pydantic model)
        use_structured = response_format is not None and hasattr(response_format, 'model_fields')
        
        # Reasoning models (GPT-5, O1, O3) have different API requirements
        is_reasoning = self._is_reasoning_model()
        
        # Build common parameters for reasoning vs standard models
        if is_reasoning:
            # Reasoning models need more tokens (reasoning + output)
            adjusted_tokens = self._get_min_tokens_for_reasoning(max_tokens)
            if adjusted_tokens > max_tokens:
                logger.info(f"Reasoning model {self.model_path}: increasing tokens from {max_tokens} to {adjusted_tokens} "
                           f"(reasoning models need extra tokens for internal reasoning)")
            
            # Reasoning models: use max_completion_tokens, don't pass temperature (uses default of 1.0)
            common_params = {"max_completion_tokens": adjusted_tokens}
            if temperature != 0.0 and temperature != 1.0:
                logger.warning(f"Model {self.model_path} only supports default temperature (1.0). "
                             f"Ignoring requested temperature={temperature}")
        else:
            # Standard models: use max_tokens and temperature
            common_params = {"max_tokens": max_tokens, "temperature": temperature}
        
        # Limit concurrent requests based on provider type
        MAX_CONCURRENT = self._get_max_concurrent_requests()
        
        async def _generate_single_async(prompt: str, semaphore: asyncio.Semaphore, max_retries: int = 5):
            """Generate a single response with retry logic under semaphore."""
            async with semaphore:
                last_error = None
                for attempt in range(max_retries):
                    try:
                        if use_structured:
                            completion = await self.async_client.beta.chat.completions.parse(
                                model=self.model_path,
                                messages=[{"role": "user", "content": prompt}],
                                response_format=response_format,
                                **common_params,
                            )
                        elif json_mode:
                            completion = await self.async_client.chat.completions.create(
                                model=self.model_path,
                                messages=[
                                    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                                    {"role": "user", "content": prompt}
                                ],
                                response_format={"type": "json_object"},
                                **common_params,
                            )
                        else:
                            completion = await self.async_client.chat.completions.create(
                                model=self.model_path,
                                messages=[{"role": "user", "content": prompt}],
                                **common_params,
                            )
                        return completion
                    except (APIConnectionError, APITimeoutError) as e:
                        last_error = e
                        wait_time = min(2 ** attempt + 1, 60)
                        logger.warning(f"Connection error on attempt {attempt + 1}/{max_retries}: {type(e).__name__}. "
                                       f"Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    except APIStatusError as e:
                        if e.status_code in (429, 500, 502, 503, 504):  # Retryable errors
                            last_error = e
                            wait_time = min(2 ** attempt + 1, 60)
                            logger.warning(f"Server error {e.status_code} on attempt {attempt + 1}/{max_retries}. "
                                           f"Retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        else:
                            logger.error(f"API error: {e.status_code}: {e}")
                            return e
                    except Exception as e:
                        logger.error(f"Unexpected error: {type(e).__name__}: {e}")
                        return e
                
                logger.error(f"All {max_retries} retries exhausted. Last error: {type(last_error).__name__}: {last_error}")
                return last_error
        
        async def _batch_generate():
            semaphore = asyncio.Semaphore(MAX_CONCURRENT)
            tasks = [_generate_single_async(prompt, semaphore) for prompt in prompts]
            return await asyncio.gather(*tasks)
        
        # Run async batch
        completions = asyncio.run(_batch_generate())
        
        responses = []
        for completion in completions:
            if isinstance(completion, Exception):
                logger.error(f"Error in OpenAI API call: {completion}")
                if use_structured:
                    responses.append({"error": str(completion)})
                else:
                    responses.append("")
            else:
                if use_structured:
                    # Return parsed Pydantic object
                    parsed = completion.choices[0].message.parsed
                    if parsed is not None:
                        responses.append(parsed)
                    else:
                        # Handle refusal or parsing failure
                        refusal = completion.choices[0].message.refusal
                        responses.append({"error": "Couldn't parse output", "refusal": refusal})
                else:
                    text = completion.choices[0].message.content
                    responses.append(self._postprocess_output(text))
        
        return responses

    def _generate_openai_compatible(
        self, 
        prompts: List[str], 
        max_tokens: int, 
        temperature: float
    ) -> List[str]:
        """Generate using OpenAI-compatible API (NVIDIA NIM, etc.) with concurrency limits."""
        from openai import APIConnectionError, APITimeoutError, APIStatusError
        
        model_name = self.config.model_name or self.model_path
        MAX_CONCURRENT = self._get_max_concurrent_requests()
        
        async def _generate_single_async(prompt: str, semaphore: asyncio.Semaphore, max_retries: int = 5):
            """Generate a single response with retry logic under semaphore."""
            async with semaphore:
                last_error = None
                for attempt in range(max_retries):
                    try:
                        completion = await self.async_client.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        return completion
                    except (APIConnectionError, APITimeoutError) as e:
                        last_error = e
                        wait_time = min(2 ** attempt + 1, 60)
                        logger.warning(f"Connection error on attempt {attempt + 1}/{max_retries}: {type(e).__name__}. "
                                       f"Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    except APIStatusError as e:
                        if e.status_code in (429, 500, 502, 503, 504):
                            last_error = e
                            wait_time = min(2 ** attempt + 1, 60)
                            logger.warning(f"Server error {e.status_code} on attempt {attempt + 1}/{max_retries}. "
                                           f"Retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        else:
                            logger.error(f"API error: {e.status_code}: {e}")
                            return e
                    except Exception as e:
                        logger.error(f"Unexpected error: {type(e).__name__}: {e}")
                        return e
                
                logger.error(f"All {max_retries} retries exhausted. Last error: {type(last_error).__name__}: {last_error}")
                return last_error
        
        async def _batch_generate():
            semaphore = asyncio.Semaphore(MAX_CONCURRENT)
            tasks = [_generate_single_async(prompt, semaphore) for prompt in prompts]
            return await asyncio.gather(*tasks)
        
        completions = asyncio.run(_batch_generate())
        
        responses = []
        for completion in completions:
            if isinstance(completion, Exception):
                logger.error(f"Error in API call: {completion}")
                responses.append("")
            else:
                text = completion.choices[0].message.content
                responses.append(self._postprocess_output(text))
        
        return responses

    def _generate_vllm_server(
        self, 
        prompts: List[str], 
        max_tokens: int, 
        temperature: float
    ) -> List[str]:
        """Generate using vLLM server (OpenAI-compatible API) with concurrency limits.
        
        This method connects to a running vLLM server and uses its OpenAI-compatible
        API endpoint for generation. vLLM servers can handle concurrent requests
        efficiently with internal batching.
        """
        from openai import APIConnectionError, APITimeoutError, APIStatusError
        
        # Use the actual model name (the one loaded on the server)
        model_name = self.actual_model_name
        
        # vLLM servers can handle concurrent requests, but we limit to avoid overwhelming
        MAX_CONCURRENT = self._get_max_concurrent_requests()
        
        async def _generate_single_async(prompt: str, semaphore: asyncio.Semaphore, max_retries: int = 5):
            """Generate a single response with retry logic under semaphore."""
            async with semaphore:
                last_error = None
                for attempt in range(max_retries):
                    try:
                        completion = await self.async_client.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        return completion
                    except (APIConnectionError, APITimeoutError) as e:
                        last_error = e
                        wait_time = min(2 ** attempt + 1, 60)
                        logger.warning(f"Connection error on attempt {attempt + 1}/{max_retries}: {type(e).__name__}. "
                                       f"Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    except APIStatusError as e:
                        if e.status_code in (429, 500, 502, 503, 504):
                            last_error = e
                            wait_time = min(2 ** attempt + 1, 60)
                            logger.warning(f"Server error {e.status_code} on attempt {attempt + 1}/{max_retries}. "
                                           f"Retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        else:
                            logger.error(f"API error in vLLM server call: {e.status_code}: {e}")
                            return e
                    except Exception as e:
                        logger.error(f"Non-retryable error in vLLM server call: {type(e).__name__}: {e}")
                        return e
                
                logger.error(f"All {max_retries} retries exhausted. Last error: {type(last_error).__name__}: {last_error}")
                return last_error
        
        async def _batch_generate():
            semaphore = asyncio.Semaphore(MAX_CONCURRENT)
            tasks = [_generate_single_async(prompt, semaphore) for prompt in prompts]
            return await asyncio.gather(*tasks)
        
        completions = asyncio.run(_batch_generate())
        
        responses = []
        for i, completion in enumerate(completions):
            if isinstance(completion, Exception):
                logger.error(f"Error in vLLM server call for prompt {i}: {type(completion).__name__}: {completion}")
                responses.append("")
            else:
                text = completion.choices[0].message.content
                responses.append(self._postprocess_output(text))
        
        return responses
    
    # Convenience methods
    def __call__(
        self, 
        prompts: List[str], 
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> List[str]:
        """Shorthand for generate()."""
        return self.generate(prompts, max_tokens=max_tokens, temperature=temperature)
    
    def interact(
        self, 
        prompt: str, 
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs
    ) -> Any:
        """Single prompt interaction.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments:
                - json_mode (bool): Request JSON object response (OpenAI only)
                - response_format: Pydantic model for structured output (OpenAI only)
                - history: Conversation history (not fully supported)
                - system_prompt: System prompt (not fully supported)
        
        Returns:
            Generated response (string or parsed Pydantic object for structured output)
        """
        json_mode = kwargs.get('json_mode', False)
        response_format = kwargs.get('response_format')
        
        # Log warning for unsupported kwargs
        supported_kwargs = {'history', 'json_mode', 'response_format', 'system_prompt'}
        unsupported = set(kwargs.keys()) - supported_kwargs
        if unsupported:
            logger.warning(f"Unsupported arguments ignored: {unsupported}")
        
        # Handle structured output for OpenAI
        if (json_mode or response_format) and self.provider == ModelProvider.OPENAI:
            return self._generate_openai(
                [prompt], 
                max_tokens=max_tokens, 
                temperature=temperature,
                json_mode=json_mode,
                response_format=response_format
            )[0]
        
        # Warn if structured output requested for non-OpenAI provider
        if json_mode or response_format:
            if self.provider != ModelProvider.OPENAI:
                logger.warning(f"Structured output (json_mode/response_format) not supported for {self.provider.value}. Using standard generation.")
        
        # Handle history (not fully supported, just log)
        if kwargs.get('history'):
            logger.debug("History parameter not fully supported in UnifiedModelClient")
        
        return self.generate([prompt], max_tokens=max_tokens, temperature=temperature)[0]

    def batch_interact(
        self, 
        prompts: List[str], 
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs
    ) -> List[str]:
        """Batch prompt interaction.
        
        Args:
            prompts: List of input prompts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments for backward compatibility
        
        Returns:
            List of generated responses
        """
        return self.generate(prompts, max_tokens=max_tokens, temperature=temperature)


# Alias for backward compatibility  
ModelClient = UnifiedModelClient


def load_model(
    model_name: str, 
    vllm_server_url: Optional[str] = None,
    **kwargs
) -> UnifiedModelClient:
    """Load a model by name (backward compatibility wrapper).
    
    This function provides backward compatibility with the old load_model() API.
    
    Args:
        model_name: Model name
        vllm_server_url: URL of vLLM server (e.g., http://localhost:8000/v1)
                        If provided, connects to vLLM server for local model inference
        **kwargs: Additional arguments passed to UnifiedModelClient
    
    Returns:
        UnifiedModelClient instance
    """
    return UnifiedModelClient(
        model_name, 
        vllm_server_url=vllm_server_url,
        **kwargs
    )
