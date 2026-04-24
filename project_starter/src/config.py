import os

from dotenv import load_dotenv
from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    openrouter_api_key: str = Field(default="", description="OpenRouter API key")
    model_name: str = Field(default="openrouter/nvidia/nemotron-3-super-120b-a12b:free", description="The LLM model to use")
    max_steps: int = Field(default=10, description="Max steps for agent execution")
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="console", description="Logging format (json or console)")

    model_config = ConfigDict(env_file=".env", extra="ignore")


settings = Settings()

# Push the key into os.environ so LiteLLM can find it automatically
if settings.openrouter_api_key:
    os.environ["OPENROUTER_API_KEY"] = settings.openrouter_api_key
