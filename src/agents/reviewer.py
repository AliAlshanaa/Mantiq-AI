from src.core.state import AgentState
from langchain_openai import ChatOpenAI

def reviewer_node(state: AgentState):
    """
    Quality Assurance node that evaluates the draft report.
    It decides whether to approve the report or send it back for revision.
    """
    draft = state.get("draft", "")
    iteration = state.get("revision_count", 0)
    task = state.get("task", "")
    
    print(f"--- LOG: Mantiq-AI (Reviewer) is evaluating the draft (Iteration {iteration + 1}) ---")
    
    # Using temperature 0 for strict, objective evaluation
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    prompt = f"""
    You are a professional Senior Editor and Quality Assurance Critic for Mantiq-AI.
    Your goal is to ensure the report about "{task}" is perfect.

    Report to Review:
    {draft}

    Evaluation Criteria:
    1. Accuracy: Does it address the user's task correctly?
    2. Structure: Does it have an Introduction, Key Points, and Conclusion?
    3. Language: Is the Arabic professional, clear, and free of typos?
    4. Sources: Are the sources cited at the end?

    Decision Rules:
    - If the report is excellent and meets all criteria, respond ONLY with the word "APPROVE".
    - If there are any flaws or missing info, respond with "REJECT: " followed by your constructive feedback in Arabic.
    """

    # Get the reviewer's verdict
    response = llm.invoke(prompt)
    verdict = response.content.strip()

    # Logic to handle the workflow direction
    # We limit revisions to 2 to prevent infinite costs or loops
    if "APPROVE" in verdict or iteration >= 2:
        return {
            "next_step": "HUMAN",
            "revision_count": iteration + 1
        }
    else:
        return {
            "feedback": verdict,
            "next_step": "REWRITE",
            "revision_count": iteration + 1
        }
