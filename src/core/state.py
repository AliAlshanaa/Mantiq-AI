from typing import TypedDict, Annotated, List, Optional
import operator


class AgentState(TypedDict, total=False):
    """
    Enterprise State Management for Mantiq-AI.

    Design Principles:
    - total=False prevents KeyError crashes during partial state updates
    - Annotated[List[str], operator.add] enables safe accumulation across cycles
    - Optional fields allow flexible routing between nodes
    """

    # Core Task
    task: str

    # Accumulative Fields (safe for cyclic graphs)
    research_data: Annotated[List[str], operator.add]
    citations: Annotated[List[str], operator.add]

    # Draft Lifecycle
    draft: str
    feedback: str
    human_feedback: str

    # Control & Routing
    revision_count: int
    next_step: str

    # Enterprise Extensions (Optional but Recommended)
    request_id: Optional[str]          # For logging & tracing
    status: Optional[str]              # RUNNING / REVIEW / APPROVED / SAVED
    error: Optional[str]               # Error tracking