from typing import TypedDict, Annotated, List, Optional
import operator

class AgentState(TypedDict, total=False):
    # Core Task
    task: str
    
    # --- FIX: Adding selected_model to State Schema ---
    selected_model: str  # To store: 'gemini', 'openai', or 'llama'

    # Accumulative Fields
    research_data: Annotated[List[str], operator.add]
    citations: Annotated[List[str], operator.add]

    # Draft Lifecycle
    draft: str
    feedback: str
    human_feedback: str

    # Control & Routing
    revision_count: int
    next_step: str

    # Enterprise Extensions
    request_id: Optional[str]
    status: Optional[str]
    error: Optional[str]