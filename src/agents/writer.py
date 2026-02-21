from src.core.state import AgentState
from langchain_openai import ChatOpenAI

def writer_node(state: AgentState):
    # Extract data from the current state
    research_data = state["research_data"]
    task = state["task"]
    
    print(f"--- LOG: Mantiq-AI (Writer) is drafting the report for: {task} ---")
    
    # Prepare the context by joining research findings
    context = "\n".join(research_data)
    
    # Define the system prompt for the writer agent
    prompt = f"""
    You are an expert report writer for the "Mantiq-AI" system.
    Your mission is to write a detailed and organized report based on the following research data:
    {context}
    
    Report Objective: {task}
    
    Writing Requirements:
    1. Use professional Arabic language.
    2. Structure the report into: Introduction, Key Points, and Conclusion.
    3. List the sources used at the end of the report.
    """
    
    # Initialize the LLM (ensure OPENAI_API_KEY is in your .env)
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    # Invoke the model to generate the report
    response = llm.invoke(prompt)
   
   # Return the updated state
    return {
        "draft": response.content,  # Changed from final_output to draft
        "next_step": "END"
    }