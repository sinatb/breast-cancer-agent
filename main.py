from agent.Orchestrator import Orchestrator

if __name__ == "__main__":
    orch = Orchestrator()

    print(orch.invoke("List available models."))
    input("Press Enter...")

    print(orch.invoke("Check budget."))
    input("Press Enter...")

    print(orch.invoke(
        "Run ONE initial experiment using MobileNet with LINEAR_PROBE, "
        "lr=1e-4, epochs=5."
    ))

    while orch.budget.can_conduct_experiment():
        cont = input("\nContinue? (y/n): ").strip().lower()
        if cont != "y":
            break

        print(orch.invoke(
            "Analyze previous results and decide the next SINGLE experiment."
        ))

    print("\nSession complete.")
