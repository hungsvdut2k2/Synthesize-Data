from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    GOOGLE_COOKIES_DIR: str
    HUGGINGFACE_TOKEN_DIR: str
    DEEPINFRA_API_KEY_DIR: str


settings = Settings(_env_file=".env")
