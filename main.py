from agent.Orchestrator import Orchestrator
import codecs

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.cpp_extension")
if __name__ == "__main__":
    orch = Orchestrator()

    print(orch.invoke("List available models."))
    input("Press Enter...")

    print(orch.invoke("Check budget."))
    input("Press Enter...")

    while orch.budget.can_conduct_experiment():
        cont = input("\nContinue? (y/n): ").strip().lower()
        if cont != "y":
            break

        print(orch.invoke(
            "Analyze previous results and decide the next SINGLE experiment. if needed check the budget and get a summary of previous experiments."
        ))

    with codecs.open("history.txt", "w","UTF-8") as f:
        f.write(str(orch.chat_history))
    print("\nSession complete.")
