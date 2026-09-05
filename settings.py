from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_path: str = "weights/convnext_tiny_ecg.pth"
    max_upload_mb: int = 8


settings = Settings()
