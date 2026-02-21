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
        "task": "تحدث عن خطة السعودية 2030",
        "research_data": [],
        "draft": "",
        "feedback": "",
        "revision_count": 0,
        "next_step": ""
    }

   # 3. Execute the Graph
    print("--- Mantiq-AI: Starting Execution ---")
    
    for output in app.stream(initial_state):
        for key, value in output.items():
            print(f"\n[Node: {key}]")
            
            # 1. Show Research Logs
            if "research_data" in value and value["research_data"]:
                print(f"✅ Success: Researcher found {len(value['research_data'])} resources.")
            
           # 2. Show the REPORT from the Writer
            if "draft" in value and value["draft"]:
                print("\n" + "="*50)
                print("📝 MANTIQ-AI REPORT DRAFT")
                print("="*50)
                print(value["draft"])
                print("="*50)

    print("\n--- Mantiq-AI: Execution Finished ---")

if __name__ == "__main__":
    run_mantiq()