from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from src.core.state import AgentState
from dotenv import load_dotenv
load_dotenv()
search_tool = TavilyClient()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def researcher_node(state: AgentState):
    task = state["task"]
    print(f"--- LOG: Mantiq-AI (Researcher) started task: {task} ---")
    
    try:
        # Execute search
        response = search_tool.search(query=task, max_results=5)
        search_results = response.get("results", [])

        findings = []
        for result in search_results:
            content = result.get("content", "")
            url = result.get("url", "No Source")
            findings.append(f"Source: {url}\nContent: {content}")

        if not findings:
            findings = ["No specific information found on this topic."]

        return {
            "research_data": findings,
            "next_step": "Writer"
        }

    except Exception as e:
        print(f"--- ERROR: Researcher Node failed: {str(e)} ---")
        return {
            "research_data": [f"Error during research: {str(e)}"],
            "next_step": "Writer"
        }