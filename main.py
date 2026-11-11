import argparse
from pathlib import Path

from agent.Orchestrator import DEFAULT_MAX_EXPERIMENTS, DEFAULT_MAX_STEPS, AgentRunResult, Orchestrator
from agent.smoke import SMOKE_SAMPLES_PER_CLASS, smoke_llm, smoke_passed
from config import load_settings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fine-tuning agent. Unattended unless --approve is given.")
    parser.add_argument("--approve", action="store_true",
                        help="ask for confirmation before each experiment; answering no ends the session")
    parser.add_argument("--max-experiments", type=int, default=DEFAULT_MAX_EXPERIMENTS,
                        help="experiment budget (default: %(default)s)")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                        help="maximum number of LLM replies before the session is stopped (default: %(default)s)")
    parser.add_argument("--smoke", action="store_true",
                        help="check the setup: a scripted stand-in for the LLM trains two models for two epochs "
                             "on a small image sample; needs no API key")
    return parser.parse_args(argv)


def ask_approval(experiment: dict) -> bool:
    print(f"\nThe agent wants to run: {experiment}")
    return input("Run it? (y/n): ").strip().lower() == "y"


def run_session(orchestrator: Orchestrator, history_path: Path) -> AgentRunResult:
    try:
        return orchestrator.run()
    finally:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(str(orchestrator.chat_history), encoding="utf-8")
        print(f"History written to {history_path}")


if __name__ == "__main__":
    args = parse_args()
    settings = load_settings()
    orch = Orchestrator(
        settings,
        max_experiments=args.max_experiments,
        max_steps=args.max_steps,
        approve=ask_approval if args.approve else None,
        llm=smoke_llm() if args.smoke else None,
        samples_per_class=SMOKE_SAMPLES_PER_CLASS if args.smoke else None,
    )
    result = run_session(orch, settings.model_dir / "history.txt")
    print(f"\nSession complete: {result.stop_reason} "
          f"({result.experiments} experiments, {result.steps} steps).")
    if args.smoke:
        if not smoke_passed(result):
            raise SystemExit("Smoke run FAILED: not every experiment completed. See the errors above.")
        print("Smoke run passed.")
