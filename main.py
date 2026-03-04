import os
from uuid import uuid4
from dotenv import load_dotenv
from langgraph.types import Command
from src.core.graph import app
from src.database.vector_store import initialize_local_vector_db
load_dotenv()

def run_mantiq():
    """
    Main entry point for Mantiq-AI.
    Orchestrates the hybrid RAG flow with Human-in-the-loop capability.
    """
    # Initialize the state with the enterprise task
    initial_state = {
        "task": "تقرير مفصل حول التحديث عن خطة السعودية 2030",
        "research_data": [],
        "citations": [],
        "draft": "",
        "feedback": "",
        "human_feedback": "",
        "revision_count": 0,
        "next_step": ""
    }

    print("--- Mantiq-AI: Starting Enterprise Hybrid RAG Execution ---")
    
    thread_id = f"mantiq-{uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}
    pending_input = initial_state

    while True:
        interrupted = False
        # Streaming the graph for real-time observability
        for output in app.stream(pending_input, config=config):
            if "__interrupt__" in output:
                interrupted = True
                print("\n--- HITL: Human Review Required ---")
                
                decision = input("\nDecision ('approve' to save, 'rewrite' to edit): ").strip().lower()
                feedback = ""
                if decision != "approve":
                    feedback = input("Enter your feedback for the writer: ").strip()

                # Resuming the workflow with human input
                pending_input = Command(resume={
                    "decision": decision,
                    "feedback": feedback
                })
                break

            # Logging Node Transitions
            for node_name, state_update in output.items():
                print(f"\n[Node: {node_name.upper()}]")
                if "research_data" in state_update:
                    print(f"✅ Context retrieved from Local RAG & Web.")
                if "draft" in state_update:
                    print(f"📝 Draft updated (Length: {len(state_update['draft'])} chars)")

        if not interrupted:
            break

    # Final Execution Summary
    final_state = app.get_state(config).values
    print("\n" + "="*60)
    print("🏆 PROCESS COMPLETED SUCCESSFULLY")
    print("="*60)
    if final_state.get("draft"):
        print("\n📄 FINAL REPORT SNIPPET:")
        print(final_state["draft"][:1000] + "...")

if __name__ == "__main__":
    
    print("--- LOG: Initializing Vector Database... ---")
    initialize_local_vector_db()
    print("--- LOG: Vector Database Ready! ---")
    run_mantiq()