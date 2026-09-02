"""Central configuration for JARVIS Core.

Settings are loaded from environment variables / a `.env` file. Nothing here
is tied to a specific LLM backend or model: swapping FAST_MODEL_BACKEND or
BRAIN_MODEL_BACKEND is enough to change runtime behaviour without touching
API, routing, or Android-facing contracts.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROTOCOL_VERSION = "1"
SERVER_VERSION = "0.1.0"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SYSTEM_PROMPT_PATH = REPO_ROOT / "config" / "prompts" / "jarvis_system.txt"

ModelBackend = Literal["fake", "ollama", "llamacpp"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Server -----------------------------------------------------
    server_host: str = Field(default="127.0.0.1", alias="SERVER_HOST")
    server_port: int = Field(default=8000, alias="SERVER_PORT")
    development_mode: bool = Field(default=True, alias="DEVELOPMENT_MODE")
    allow_remote_connections: bool = Field(default=False, alias="ALLOW_REMOTE_CONNECTIONS")

    # --- Security -----------------------------------------------------
    api_token: str | None = Field(default=None, alias="API_TOKEN")
    max_request_body_bytes: int = Field(default=1_000_000, alias="MAX_REQUEST_BODY_BYTES")
    rate_limit_per_minute: int = Field(default=120, alias="RATE_LIMIT_PER_MINUTE")

    # --- Request handling -----------------------------------------------------
    request_timeout: float = Field(default=120.0, alias="REQUEST_TIMEOUT")
    max_concurrent_requests: int = Field(default=2, alias="MAX_CONCURRENT_REQUESTS")
    max_queue_size: int = Field(default=16, alias="MAX_QUEUE_SIZE")

    # --- Conversation / context -----------------------------------------------------
    max_conversation_messages: int = Field(default=20, alias="MAX_CONVERSATION_MESSAGES")
    max_context_chars: int = Field(default=12_000, alias="MAX_CONTEXT_CHARS")

    # --- FAST model -----------------------------------------------------
    fast_model_backend: ModelBackend = Field(default="fake", alias="FAST_MODEL_BACKEND")
    fast_model_name: str = Field(default="fast-fake", alias="FAST_MODEL_NAME")
    fast_model_path: str | None = Field(default=None, alias="FAST_MODEL_PATH")
    fast_context_size: int = Field(default=4096, alias="FAST_CONTEXT_SIZE")
    fast_threads: int = Field(default=4, alias="FAST_THREADS")
    fast_gpu_layers: int = Field(default=0, alias="FAST_GPU_LAYERS")
    fast_base_url: str = Field(default="http://127.0.0.1:11434", alias="FAST_BASE_URL")

    # --- BRAIN model -----------------------------------------------------
    brain_model_backend: ModelBackend = Field(default="fake", alias="BRAIN_MODEL_BACKEND")
    brain_model_name: str = Field(default="brain-fake", alias="BRAIN_MODEL_NAME")
    brain_model_path: str | None = Field(default=None, alias="BRAIN_MODEL_PATH")
    brain_context_size: int = Field(default=8192, alias="BRAIN_CONTEXT_SIZE")
    brain_threads: int = Field(default=4, alias="BRAIN_THREADS")
    brain_gpu_layers: int = Field(default=0, alias="BRAIN_GPU_LAYERS")
    brain_base_url: str = Field(default="http://127.0.0.1:11434", alias="BRAIN_BASE_URL")

    # --- Routing heuristics -----------------------------------------------------
    router_fast_max_chars: int = Field(default=220, alias="ROUTER_FAST_MAX_CHARS")
    router_brain_keywords: str = Field(
        default="perché,pianifica,analizza,confronta,spiega in dettaglio,ragiona,progetta,strategia,why,plan,analyze,compare,design,reasoning",
        alias="ROUTER_BRAIN_KEYWORDS",
    )
    router_fast_keywords: str = Field(
        default="ciao,ok,grazie,che ore sono,accendi,spegni,hi,hello,thanks",
        alias="ROUTER_FAST_KEYWORDS",
    )

    # --- Prompts -----------------------------------------------------
    system_prompt_path: str = Field(
        default=str(DEFAULT_SYSTEM_PROMPT_PATH), alias="SYSTEM_PROMPT_PATH"
    )

    # --- Logging -----------------------------------------------------
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_json: bool = Field(default=True, alias="LOG_JSON")

    @field_validator("max_concurrent_requests")
    @classmethod
    def _min_one_concurrent(cls, v: int) -> int:
        return max(1, v)

    @property
    def router_brain_keyword_list(self) -> list[str]:
        return [k.strip().lower() for k in self.router_brain_keywords.split(",") if k.strip()]

    @property
    def router_fast_keyword_list(self) -> list[str]:
        return [k.strip().lower() for k in self.router_fast_keywords.split(",") if k.strip()]

    def load_system_prompt(self) -> str:
        path = Path(self.system_prompt_path)
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        return "You are JARVIS, a helpful personal AI assistant."


@lru_cache
def get_settings() -> Settings:
    return Settings()
