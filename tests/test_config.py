"""Configuration is read when a run starts, never at import time (C6)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from config import PROJECT_ROOT, ConfigError, load_settings


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """A folder shaped like the real dataset, with no real data in it."""
    folder = tmp_path / "data"
    (folder / "images").mkdir(parents=True)
    (folder / "train_labels.csv").write_text("filename,label\n")
    return folder


def _write_env(path: Path, **values: str) -> Path:
    path.write_text("".join(f"{k}={v}\n" for k, v in values.items()))
    return path


# --------------------------------------------------------------------------- import time


def test_modules_import_without_configuration(tmp_path):
    """C6: a clean clone has no .env, and importing the project must not crash because of it."""
    env = {k: v for k, v in os.environ.items() if k not in {"DATASET_PATH", "MODEL_DIR", "API_KEY", "BASE_URL"}}
    # Stub python-dotenv so no .env file anywhere can supply the settings.
    code = (
        "import sys, types\n"
        "stub = types.ModuleType('dotenv')\n"
        "stub.load_dotenv = lambda *a, **k: False\n"
        "stub.dotenv_values = lambda *a, **k: {}\n"
        "sys.modules['dotenv'] = stub\n"
        "import dataset.DataLoader, finetuning.FineTuner, agent.tools, agent.Orchestrator\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**env, "PYTHONPATH": str(PROJECT_ROOT)},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]


# --------------------------------------------------------------------------- load_settings


def test_reads_settings_from_env_file(tmp_path, dataset_dir):
    env_file = _write_env(
        tmp_path / ".env",
        DATASET_PATH=str(dataset_dir),
        MODEL_DIR=str(tmp_path / "runs"),
        API_KEY="secret",
        BASE_URL="https://llm.example/v1",
        LLM_MODEL="some-model",
    )

    settings = load_settings(env_file=env_file, environ={})

    assert settings.dataset_path == dataset_dir
    assert settings.images_dir == dataset_dir / "images"
    assert settings.labels_csv == dataset_dir / "train_labels.csv"
    assert settings.model_dir == tmp_path / "runs"
    assert settings.api_key == "secret"
    assert settings.base_url == "https://llm.example/v1"
    assert settings.llm_model == "some-model"


def test_environment_overrides_env_file(tmp_path, dataset_dir):
    env_file = _write_env(tmp_path / ".env", DATASET_PATH="/does/not/exist", MODEL_DIR="from-file")

    settings = load_settings(
        env_file=env_file, environ={"DATASET_PATH": str(dataset_dir), "MODEL_DIR": str(tmp_path / "env")}
    )

    assert settings.dataset_path == dataset_dir
    assert settings.model_dir == tmp_path / "env"


def test_missing_env_file_is_fine_when_environment_has_everything(tmp_path, dataset_dir):
    settings = load_settings(
        env_file=tmp_path / "absent.env", environ={"DATASET_PATH": str(dataset_dir), "MODEL_DIR": "runs"}, root=tmp_path
    )
    assert settings.dataset_path == dataset_dir


def test_relative_paths_resolve_against_root(tmp_path, dataset_dir):
    settings = load_settings(
        env_file=tmp_path / "absent.env", environ={"DATASET_PATH": "data", "MODEL_DIR": "./runs/"}, root=tmp_path
    )
    assert settings.dataset_path == dataset_dir
    assert settings.model_dir == tmp_path / "runs"


def test_defaults_for_optional_settings(tmp_path, dataset_dir):
    settings = load_settings(
        env_file=tmp_path / "absent.env", environ={"DATASET_PATH": str(dataset_dir), "MODEL_DIR": "runs"}, root=tmp_path
    )
    assert settings.api_key is None
    assert settings.base_url is None
    assert settings.llm_model == "gpt-4.1-mini"


def test_missing_required_settings_are_all_named(tmp_path):
    with pytest.raises(ConfigError) as err:
        load_settings(env_file=tmp_path / "absent.env", environ={})
    assert "DATASET_PATH" in str(err.value)
    assert "MODEL_DIR" in str(err.value)


def test_empty_value_counts_as_missing(tmp_path, dataset_dir):
    with pytest.raises(ConfigError, match="MODEL_DIR"):
        load_settings(env_file=tmp_path / "absent.env", environ={"DATASET_PATH": str(dataset_dir), "MODEL_DIR": " "})


def test_dataset_path_must_exist(tmp_path):
    with pytest.raises(ConfigError, match="DATASET_PATH"):
        load_settings(
            env_file=tmp_path / "absent.env", environ={"DATASET_PATH": str(tmp_path / "nope"), "MODEL_DIR": "r"}
        )


@pytest.mark.parametrize("missing", ["images", "train_labels.csv"])
def test_dataset_path_must_have_images_and_labels(tmp_path, dataset_dir, missing):
    target = dataset_dir / missing
    if target.is_dir():
        target.rmdir()
    else:
        target.unlink()

    with pytest.raises(ConfigError, match=missing):
        load_settings(env_file=tmp_path / "absent.env", environ={"DATASET_PATH": str(dataset_dir), "MODEL_DIR": "r"})


def test_require_llm_names_missing_api_key(tmp_path, dataset_dir):
    settings = load_settings(
        env_file=tmp_path / "absent.env", environ={"DATASET_PATH": str(dataset_dir), "MODEL_DIR": "runs"}, root=tmp_path
    )
    with pytest.raises(ConfigError, match="API_KEY"):
        settings.require_llm()


def test_require_llm_passes_with_api_key(tmp_path, dataset_dir):
    settings = load_settings(
        env_file=tmp_path / "absent.env",
        environ={"DATASET_PATH": str(dataset_dir), "MODEL_DIR": "runs", "API_KEY": "k"},
        root=tmp_path,
    )
    settings.require_llm()
