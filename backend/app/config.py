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

    # ToltIQ -- Phase 4 ad-hoc query against built data rooms. Read at
    # runtime by services/toltiq_adhoc.py via os.environ; declared here
    # only so pydantic-settings doesn't whine on the .env load. Blank
    # values cause the chat tool to surface a friendly "not configured"
    # message.
    toltiq_base_url: str = ""
    toltiq_api_key: str = ""
    toltiq_org_id: str = ""

    @property
    def origin_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


settings = Settings()
