from src.core.state import AgentState
from langchain_openai import ChatOpenAI

def writer_node(state: AgentState):
    """
    Writer Node: Synthesizes research and embeds academic citations.
    """
    research_data = state["research_data"]
    citations = state.get("citations", [])
    task = state["task"]
    
    print(f"--- LOG: Mantiq-AI (Writer) drafting report with citations ---")
    
    context = "\n".join(research_data)
    # Create a numbered list of references for the prompt
    references_str = "\n".join([f"[{i+1}] {cite}" for i, cite in enumerate(citations)])
    
    prompt = f"""
    You are a professional report writer. Create a detailed report about: {task}
    
    Research Data:
    {context}
    
    Available Citations (APA):
    {references_str}
    
    Requirements:
    1. Language: Use professional Arabic.
    2. In-text Citations: You MUST use numbers like [1] or [2] after every claim based on the sources.
    3. Structure: Introduction, Analysis, and a final "References" (المراجع) section.
    4. References Section: List all citations in the provided APA format at the end.
    """
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    response = llm.invoke(prompt)
   
    return {
        "draft": response.content,
        "next_step": "REVIEWER"
    }