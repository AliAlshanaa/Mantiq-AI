import os
from dotenv import load_dotenv
from src.core.graph import app

# 1. Initialize environment variables from .env file
load_dotenv()

def run_mantiq():
    """
    Main execution function for Mantiq-AI orchestration.
    This manages the flow between Researcher, Writer, and Reviewer.
    """
    
    # 2. Define the starting state for the Agentic Workflow
    initial_state = {
        "task": "تحدث عن خطة السعودية 2030",
        "research_data": [],
        "draft": "",
        "feedback": "",
        "revision_count": 0,
        "next_step": ""
    }

    print("--- Mantiq-AI: Starting Agentic Execution ---")
    
    # 3. Stream the graph execution to monitor node transitions in real-time
    final_output_state = None
    
    for output in app.stream(initial_state):
        # The output is a dictionary where keys are node names
        for key, value in output.items():
            print(f"\n[Node: {key}]")
            
            # Capture the state to access it outside the loop if needed
            final_output_state = value 
            
            # --- RESEARCHER LOGS ---
            if "research_data" in value and value["research_data"]:
                print(f"✅ Success: Researcher gathered {len(value['research_data'])} data points.")
            
            # --- WRITER LOGS ---
            if "draft" in value and value["draft"]:
                print("\n📝 NEW DRAFT GENERATED")
                # To keep terminal clean, we print a snippet during streaming
                print(f"Content Preview: {value['draft'][:150]}...")

            # --- REVIEWER LOGS ---
            if "feedback" in value and value["feedback"]:
                if "APPROVE" in value["feedback"]:
                    print("⭐ REVIEWER STATUS: Draft Approved!")
                else:
                    print(f"⚠️ FEEDBACK FROM REVIEWER:\n{value['feedback']}")

    # 4. Final Output Display after all iterations are complete
    print("\n" + "="*60)
    print("🏆 FINAL MANTIQ-AI PROCESS COMPLETED")
    print("="*60)
    
    # Note: In LangGraph stream, we get the final state from the last node
    # Let's invoke one last time to get the absolute final state object
    final_state = app.invoke(initial_state)
    
    if final_state.get("draft"):
        print("\n✨ APPROVED REPORT:")
        print(final_state["draft"])
    
    print("\n--- Mantiq-AI: Termination Signal Received ---")

if __name__ == "__main__":
    run_mantiq()