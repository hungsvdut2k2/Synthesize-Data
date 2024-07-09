from pydantic import BaseModel, Field


class SynthesizeParameters(BaseModel):
    system_prompt: str = Field(default=None)
    user_prompt: str = Field(default=None)
    model_name: str = Field(default="default")
    api_key: str = Field(default=None)
