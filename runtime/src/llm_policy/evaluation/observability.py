"""Optional Langfuse tracing for policy-generation runs."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


@dataclass
class LangfuseProviderClient:
    """Add one Langfuse generation observation around each provider call."""

    provider: Any
    langfuse: Any
    call_index: int = 0

    def ensure_model_available(self, model: str) -> None:
        """Delegate the free model-catalog preflight without creating a span."""
        self.provider.ensure_model_available(model)

    def complete(self, *, model, messages, max_tokens) -> Any:
        """Trace prompt, response, model parameters, errors, and token usage."""
        call_index = self.call_index
        self.call_index += 1
        with self.langfuse.start_as_current_observation(
            as_type="generation",
            name=f"openrouter-policy-attempt-{call_index}",
            model=model,
            input=messages,
            model_parameters={"max_tokens": int(max_tokens)},
            metadata={"provider": "openrouter", "attempt": call_index},
        ) as generation:
            try:
                response = self.provider.complete(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                )
            except Exception as error:
                generation.update(
                    level="ERROR",
                    status_message=f"{type(error).__name__}: {error}",
                )
                raise

            usage_details = None
            if response.usage is not None:
                usage_details = {
                    "input": response.usage.input_tokens,
                    "output": response.usage.completion_tokens,
                    "total": response.usage.total_tokens,
                    "reasoning": response.usage.reasoning_tokens,
                }
            generation.update(
                output=response.content,
                usage_details=usage_details,
                metadata={
                    "provider": "openrouter",
                    "attempt": call_index,
                    "response_id": response.raw.get("id"),
                },
            )
            return response


@contextmanager
def traced_policy_generation(config, prompt, environment, provider):
    """Create one root trace and yield its traced provider plus trace ID."""
    from langfuse import get_client

    langfuse = get_client()
    if not langfuse.auth_check():
        raise RuntimeError("Langfuse authentication failed")
    try:
        with langfuse.start_as_current_observation(
            as_type="agent",
            name="llm-joint-policy-generation",
            input={
                "prompt_hash": prompt.prompt_hash,
                "environment": environment.env_name,
                "environment_config": environment.config,
                "model": config.model,
                "token_budget": config.token_budget,
            },
            metadata={
                "env_config": environment.config_path.name,
                "max_attempts": config.max_attempts,
                "num_eval_envs": config.num_eval_envs,
                "eval_seed": config.eval_seed,
            },
        ) as root:
            trace_id = root.trace_id
            traced_provider = LangfuseProviderClient(provider, langfuse)
            yield traced_provider, trace_id, root
    finally:
        langfuse.flush()


__all__ = ["LangfuseProviderClient", "traced_policy_generation"]
