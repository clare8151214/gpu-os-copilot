import os
from typing import Annotated, Literal
from urllib.parse import urlparse

from pydantic import Field, model_validator

from gpu_os_copilot.schemas import Model


class AgentConfig(Model):
    model: str = Field(default="muse-spark-1.3", pattern=r"^[A-Za-z0-9_.-]{1,100}$")
    base_url: str = "https://api.meta.ai/v1"
    max_steps: Annotated[int, Field(strict=True, ge=1, le=10)] = 5
    max_experiments: Annotated[int, Field(strict=True, ge=0, le=2)] = 2
    max_tool_calls: Annotated[int, Field(strict=True, ge=1, le=20)] = 12
    max_completion_tokens: Annotated[int, Field(strict=True, ge=256, le=8192)] = 4096
    reasoning_effort: Literal["minimal", "low", "medium", "high"] = "medium"
    dry_run: bool = True

    @model_validator(mode="after")
    def official_endpoint(self):
        parsed = urlparse(self.base_url)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "api.meta.ai"
            or parsed.path.rstrip("/") != "/v1"
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Only the official HTTPS Meta endpoint is supported")
        return self

    @classmethod
    def from_environment(cls):
        return cls(
            model=os.getenv("META_MODEL", "muse-spark-1.3"),
            base_url=os.getenv("META_API_BASE_URL", "https://api.meta.ai/v1"),
            max_steps=int(os.getenv("META_MAX_AGENT_STEPS", "5")),
            max_experiments=int(os.getenv("META_MAX_EXPERIMENTS", "2")),
            max_completion_tokens=int(os.getenv("META_MAX_COMPLETION_TOKENS", "4096")),
            reasoning_effort=os.getenv("META_REASONING_EFFORT", "medium"),
            dry_run=os.getenv("DRY_RUN", "1") != "0",
        )


class CopilotError(RuntimeError):
    """A fixed public error message containing no provider response bodies or secrets."""
