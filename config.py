
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_LLM_MODEL = "gpt-4.1-mini"


class ConfigError(RuntimeError):
    """A required setting is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    dataset_path: Path
    model_dir: Path
    api_key: str | None = None
    base_url: str | None = None
    llm_model: str = DEFAULT_LLM_MODEL

    @property
    def images_dir(self) -> Path:
        return self.dataset_path / "images"

    @property
    def labels_csv(self) -> Path:
        return self.dataset_path / "train_labels.csv"

    def require_llm(self) -> None:
        if not self.api_key:
            raise ConfigError("API_KEY is not set. Add it to .env (see .env.example) or the environment.")


def load_settings(
    env_file: Path | None = None,
    environ: Mapping[str, str] | None = None,
    root: Path = PROJECT_ROOT,
) -> Settings:
    env_file = root / ".env" if env_file is None else env_file
    file_values = dotenv_values(env_file) if env_file.is_file() else {}
    environ = os.environ if environ is None else environ

    def get(name: str) -> str | None:
        for source in (environ, file_values):
            value = source.get(name)
            if value is not None and value.strip():
                return value.strip()
        return None

    missing = [name for name in ("DATASET_PATH", "MODEL_DIR") if get(name) is None]
    if missing:
        raise ConfigError(
            f"Missing required setting(s): {', '.join(missing)}. "
            f"Copy .env.example to .env and fill them in, or set them in the environment."
        )

    settings = Settings(
        dataset_path=_resolve(get("DATASET_PATH"), root),
        model_dir=_resolve(get("MODEL_DIR"), root),
        api_key=get("API_KEY"),
        base_url=get("BASE_URL"),
        llm_model=get("LLM_MODEL") or DEFAULT_LLM_MODEL,
    )
    _check_dataset(settings)
    return settings


def _resolve(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def _check_dataset(settings: Settings) -> None:
    if not settings.dataset_path.is_dir():
        raise ConfigError(f"DATASET_PATH is not a folder: {settings.dataset_path}")
    if not settings.images_dir.is_dir():
        raise ConfigError(f"DATASET_PATH has no images folder: {settings.images_dir}")
    if not settings.labels_csv.is_file():
        raise ConfigError(f"DATASET_PATH has no train_labels.csv: {settings.labels_csv}")
