"""Command-line entry point: unattended by default, and the transcript survives a crash."""

import pytest

from main import parse_args, run_session


class FakeOrchestrator:
    def __init__(self, error: Exception | None = None):
        self.chat_history = ["system prompt", "first step"]
        self.error = error

    def run(self):
        if self.error is not None:
            raise self.error
        return "result"


def test_unattended_is_the_default():
    args = parse_args([])

    assert args.approve is False
    assert args.max_experiments == 10
    assert args.max_steps > args.max_experiments


def test_flags_are_parsed():
    args = parse_args(["--approve", "--max-experiments", "2", "--max-steps", "7"])

    assert (args.approve, args.max_experiments, args.max_steps) == (True, 2, 7)


def test_history_is_written_after_a_run(tmp_path):
    history = tmp_path / "runs" / "history.txt"

    assert run_session(FakeOrchestrator(), history) == "result"
    assert "first step" in history.read_text(encoding="utf-8")


def test_history_is_written_when_the_run_crashes(tmp_path):
    history = tmp_path / "runs" / "history.txt"

    with pytest.raises(KeyboardInterrupt):
        run_session(FakeOrchestrator(KeyboardInterrupt()), history)

    assert "first step" in history.read_text(encoding="utf-8")
