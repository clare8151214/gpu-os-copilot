import os

from gpu_os_copilot.config import AgentConfig, CopilotError


class MetaClient:
    mode = "meta"

    def __init__(self, config: AgentConfig, *, allow_paid_api: bool = False):
        if not allow_paid_api:
            raise CopilotError("PAID_API_NOT_APPROVED: use --allow-paid-api for live Meta calls")
        key = os.getenv("MODEL_API_KEY")
        if not key or key == "REPLACE_ME":
            raise CopilotError("META_KEY_MISSING: export MODEL_API_KEY before a live call")
        try:
            from openai import OpenAI
        except ImportError:
            raise CopilotError("META_SDK_MISSING: install the project's meta extra") from None
        self.config = config
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=key,
            timeout=30,
            max_retries=0,
        )

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        from openai import APIError, AuthenticationError, RateLimitError

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                **({"tools": tools, "tool_choice": "auto"} if tools else {}),
                max_completion_tokens=self.config.max_completion_tokens,
                reasoning_effort=self.config.reasoning_effort,
            )
        except AuthenticationError:
            raise CopilotError(
                "META_AUTHENTICATION_FAILED: check the Meta API credential"
            ) from None
        except RateLimitError:
            raise CopilotError("META_QUOTA_OR_RATE_LIMIT: no fallback provider was used") from None
        except APIError:
            raise CopilotError(
                "META_API_FAILED: request failed; provider body suppressed"
            ) from None
        if not response.choices:
            raise CopilotError("META_EMPTY_RESPONSE")
        return {
            "message": response.choices[0].message.model_dump(exclude_none=True),
            "usage": response.usage.model_dump() if response.usage else None,
            "response_id": response.id,
            "finish_reason": response.choices[0].finish_reason,
        }
