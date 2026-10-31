"""Reserve bounded request volume and completion capacity before each provider call."""

import json

from gpu_os_copilot.config import CopilotError


class ModelBudget:
    def __init__(self, config):
        self.config = config
        self.requests = 0
        self.input_bytes = 0
        self.reserved_completion_tokens = 0
        self.responses = 0
        self.responses_with_complete_usage = 0
        self.reported_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def reserve(self, messages, tools):
        # This is an exact serialization-byte bound, not an estimate of provider tokens.
        size = len(json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False).encode())
        if size > self.config.max_request_bytes:
            raise CopilotError("AGENT_REQUEST_BYTE_BUDGET_EXCEEDED")
        if self.input_bytes + size > self.config.max_session_request_bytes:
            raise CopilotError("AGENT_SESSION_BYTE_BUDGET_EXCEEDED")
        if (
            self.reserved_completion_tokens + self.config.max_completion_tokens
            > self.config.max_total_completion_tokens
        ):
            raise CopilotError("AGENT_COMPLETION_BUDGET_EXCEEDED")
        self.input_bytes += size
        self.reserved_completion_tokens += self.config.max_completion_tokens
        self.requests += 1

    def observe(self, usage):
        self.responses += 1
        if usage is None:
            return
        if not isinstance(usage, dict):
            raise CopilotError("AGENT_INVALID_USAGE")
        known = {}
        for field in self.reported_tokens:
            value = usage.get(field)
            if value is not None:
                if type(value) is not int or value < 0:
                    raise CopilotError("AGENT_INVALID_USAGE")
                known[field] = value
        if known.get("completion_tokens", 0) > self.config.max_completion_tokens:
            raise CopilotError("AGENT_PROVIDER_COMPLETION_LIMIT_EXCEEDED")
        if len(known) == 3:
            if known["prompt_tokens"] + known["completion_tokens"] != known["total_tokens"]:
                raise CopilotError("AGENT_INVALID_USAGE")
            self.responses_with_complete_usage += 1
        for field, value in known.items():
            self.reported_tokens[field] += value

    def report(self):
        return {
            "requests_reserved": self.requests,
            "serialized_input_bytes": self.input_bytes,
            "completion_tokens_reserved": self.reserved_completion_tokens,
            "remaining_completion_reservation": self.config.max_total_completion_tokens
            - self.reserved_completion_tokens,
            "responses_received": self.responses,
            "responses_with_complete_usage": self.responses_with_complete_usage,
            "reported_provider_tokens": self.reported_tokens.copy(),
            "provider_usage_complete": self.responses > 0
            and self.responses_with_complete_usage == self.responses,
            "cost_usd": None,
            "limitations": [
                "Reserved completion capacity is not actual usage or a monetary charge.",
                "Request limits count serialized messages/tools bytes, not provider input tokens.",
                "Missing provider usage stays incomplete; no token price is assumed.",
            ],
        }

    def feedback_capacity(self):
        """Conservatively reserve a full allowed request for subsequent feedback."""
        return max(
            0,
            min(
                self.config.max_steps - self.requests,
                (self.config.max_total_completion_tokens - self.reserved_completion_tokens)
                // self.config.max_completion_tokens,
                (self.config.max_session_request_bytes - self.input_bytes)
                // self.config.max_request_bytes,
            ),
        )
