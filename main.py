import os
from dotenv import load_dotenv
from src.core.graph import app

# 1. Load Environment Variables (API Keys)
load_dotenv()

def run_mantiq():
    """
    Initial test run for Mantiq-AI
    """
    # 2. Define the initial state (The input task)
    initial_state = {
        "task": "Current status of AI adoption in Saudi Arabia's government sector 2026",
        "research_data": [],
        "draft": "",
        "feedback": "",
        "revision_count": 0,
        "next_step": ""
    }

    # 3. Execute the Graph
    print("--- Mantiq-AI: Starting Execution ---")
    
    # We use .stream to see the output of each node as it happens
    for output in app.stream(initial_state):
        for key, value in output.items():
            print(f"\n[Node: {key}]")
            # If the node produced research data, let's print a snippet
            if "research_data" in value:
                print(f"Data gathered: {len(value['research_data'])} sources found.")
                for info in value['research_data']:
                    print(f"- {info[:100]}...") # Print first 100 characters of each finding

    print("\n--- Mantiq-AI: Execution Finished ---")

if __name__ == "__main__":
    run_mantiq()