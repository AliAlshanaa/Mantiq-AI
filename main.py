from uuid import uuid4
from dotenv import load_dotenv
from langgraph.types import Command
from src.core.graph import app

# 1. Initialize environment variables from .env file
load_dotenv()

def run_mantiq():
    """
    Main execution function for Mantiq-AI orchestration.
    This manages the flow between Researcher, Writer, Reviewer, and Human Review.
    """
    
    # 2. Define the starting state for the Agentic Workflow
    initial_state = {
        "task": "    تقرير مفصل حول التحديث عن خطة السعودية 2030",
        "research_data": [],
        "draft": "",
        "feedback": "",
        "human_feedback": "",
        "revision_count": 0,
        "next_step": ""
    }

    print("--- Mantiq-AI: Starting Agentic Execution ---")
    
    # 3. Stream the graph execution to monitor node transitions in real-time
    final_output_state = None
    thread_id = f"mantiq-{uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}
    pending_input = initial_state

    while True:
        interrupted = False

        for output in app.stream(pending_input, config=config):
            if "__interrupt__" in output:
                interrupted = True
                interrupts = output.get("__interrupt__", [])
                interrupt_payload = None
                if interrupts:
                    candidate = interrupts[0]
                    interrupt_payload = getattr(candidate, "value", candidate)

                print("\n--- HITL: Human Review Required ---")
                if isinstance(interrupt_payload, dict):
                    if interrupt_payload.get("reviewer_feedback"):
                        print(f"Reviewer Feedback:\n{interrupt_payload['reviewer_feedback']}")
                    if interrupt_payload.get("draft"):
                        print("\nDraft Preview:")
                        print(interrupt_payload["draft"][:800])

                decision = input("\nType 'approve' or 'rewrite': ").strip()
                feedback = ""
                if decision.lower() not in {"approve", "approved", "yes", "y"}:
                    feedback = input("Provide feedback for the writer: ").strip()

                pending_input = Command(resume={
                    "decision": decision,
                    "feedback": feedback
                })
                break

            # The output is a dictionary where keys are node names
            for key, value in output.items():
                print(f"\n[Node: {key}]")
                
                # Capture the state to access it outside the loop if needed
                final_output_state = value 
                
                # --- RESEARCHER LOGS ---
                if "research_data" in value and value["research_data"]:
                    print(f"âœ… Success: Researcher gathered {len(value['research_data'])} data points.")
                
                # --- WRITER LOGS ---
                if "draft" in value and value["draft"]:
                    print("\nًں“‌ NEW DRAFT GENERATED")
                    # To keep terminal clean, we print a snippet during streaming
                    print(f"Content Preview: {value['draft'][:150]}...")

                # --- REVIEWER LOGS ---
                if "feedback" in value and value["feedback"]:
                    if "APPROVE" in value["feedback"]:
                        print("â­گ REVIEWER STATUS: Draft Approved!")
                    else:
                        print(f"âڑ ï¸ڈ FEEDBACK FROM REVIEWER:\n{value['feedback']}")

        if not interrupted:
            break

    # 4. Final Output Display after all iterations are complete
    print("\n" + "="*60)
    print("ًںڈ† FINAL MANTIQ-AI PROCESS COMPLETED")
    print("="*60)
    
    # Get the final state snapshot from the checkpointer
    try:
        final_state = app.get_state(config).values
    except Exception:
        final_state = final_output_state or {}
    
    if final_state.get("draft"):
        print("\nâœ¨ APPROVED REPORT:")
        print(final_state["draft"])
    
    print("\n--- Mantiq-AI: Termination Signal Received ---")

if __name__ == "__main__":
    run_mantiq()
