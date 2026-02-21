from langgraph.graph import StateGraph, END
from src.core.state import AgentState
from src.agents.researcher import researcher_node
from src.agents.writer import writer_node # Import the new writer node

# 1. Initialize the Graph with our custom State
workflow = StateGraph(AgentState)

# 2. Add Nodes
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node) # Add the Writer node to the graph

# 3. Define the Flow (Edges)
# The process starts with the Researcher
workflow.set_entry_point("researcher")

# After research is finished, the flow moves to the Writer
workflow.add_edge("researcher", "writer")

# After the Writer finishes the report, the process ends
workflow.add_edge("writer", END)

# 4. Compile the graph
app = workflow.compile()