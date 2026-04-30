"""Settings loaded from .env / environment. Single source of truth for config."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    anthropic_api_key: str = ""

    azure_client_id: str = ""
    azure_client_secret: str = ""
    azure_tenant_id: str = ""
    flask_secret_key: str = ""

    # Todd the Walrus -- Slack app credentials. Empty in local dev means
    # the Slack endpoints will refuse signed requests (good: prevents
    # accidental staging access without proper config).
    slack_signing_secret: str = ""
    slack_bot_token: str = ""

    allowed_origins: str = ""

    @property
    def origin_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


settings = Settings()
