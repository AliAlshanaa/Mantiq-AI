import os
from uuid import uuid4
from dotenv import load_dotenv
from langgraph.types import Command
from src.core.graph import app
from src.core.factory import get_model_choice  # Import the new Model Factory
from src.database.vector_store import initialize_local_vector_db

load_dotenv()

def run_mantiq():
    """
    Main entry point for Mantiq-AI.
    Orchestrates the hybrid RAG flow with Human-in-the-loop capability.
    Now supports dynamic AI model selection.
    """
    
    # --- Step 1: User Model Selection ---
    # Ask the user which 'Brain' they want to use for this session
    selected_provider = get_model_choice()

    # --- Step 2: Initialize State ---
    initial_state = {
        "task": "تقرير مفصل حول التحديث عن خطة السعودية 2030",
        "selected_model": selected_provider, # Injecting the choice into the graph state
        "research_data": [],
        "citations": [],
        "draft": "",
        "feedback": "",
        "human_feedback": "",
        "revision_count": 0,
        "next_step": ""
    }

    print(f"\n--- Mantiq-AI: Executing with [{selected_provider.upper()}] engine ---")
    
    thread_id = f"mantiq-{uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}
    pending_input = initial_state

    # --- Step 3: Graph Execution Loop ---
    while True:
        interrupted = False
        
        # Stream the graph execution for real-time visibility
        for output in app.stream(pending_input, config=config):
            
            # Check for Human-in-the-loop (HITL) interrupt
            if "__interrupt__" in output:
                interrupted = True
                print("\n--- 🛑 HITL: Human Review Required ---")
                
                decision = input("\nDecision ('approve' to finalise, 'rewrite' to provide feedback): ").strip().lower()
                feedback = ""
                if decision != "approve":
                    feedback = input("📝 Enter your instructions for the writer: ").strip()

                # Resuming the workflow using Command(resume=...)
                pending_input = Command(resume={
                    "decision": decision,
                    "feedback": feedback
                })
                break

            # Logging Logic for Node Transitions
            for node_name, state_update in output.items():
                print(f"\n[Node: {node_name.upper()}]")
                if "research_data" in state_update:
                    print("✅ Context retrieved successfully from Hybrid Sources.")
                if "draft" in state_update:
                    print(f"✍️ Draft updated (Length: {len(state_update['draft'])} characters).")

        if not interrupted:
            break

    # --- Step 4: Final Summary Output ---
    final_state = app.get_state(config).values
    print("\n" + "="*60)
    print("🏆 REPORTING PROCESS COMPLETED")
    print("="*60)
    
    if final_state.get("draft"):
        print("\n📄 FINAL REPORT PREVIEW:")
        print("-" * 30)
        print(final_state["draft"][:1500] + "...")
        print("-" * 30)
        print("✅ Full report is ready for export.")

if __name__ == "__main__":
    # Check if vector DB exists, if not, initialize it
    # This ensures RAG is ready before the agents start searching
    db_path = "./data/vectorstore"
    if not os.path.exists(db_path):
        print("--- LOG: Initializing Local Vector Database for the first time... ---")
        initialize_local_vector_db()
    else:
        print("--- LOG: Local Vector Database found and loaded. ---")
    
    try:
        run_mantiq()
    except KeyboardInterrupt:
        print("\n--- LOG: Process terminated by user. ---")
    except Exception as e:
        print(f"\n--- CRITICAL ERROR: {str(e)} ---")