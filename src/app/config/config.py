import os
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from dotenv import load_dotenv

load_dotenv()


class Config(NamedTuple):
    env: str
    project_title: str
    project_version: str
    datetime_format: str
    date_format: str
    cors_origins: list[str]
    root_path: Path
    static_path: Path
    gemini_api_key: str
    openai_api_key: str
    hugging_face_token: str


@lru_cache
def get_config() -> Config:
    root = Path(__file__).resolve().parent.parent.parent.parent

    return Config(
        env=os.getenv("ENV", "dev"),
        project_title="PhD Research Project by Dima Zharkov",
        project_version="0.0.1",
        datetime_format="%Y-%m-%dT%H:%M:%S",
        date_format="%Y-%m-%d",
        cors_origins=["*"],
        root_path=root,
        static_path=root / "static",
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        hugging_face_token=os.getenv("HUGGING_FACE_TOKEN", ""),
    )


config = get_config()
