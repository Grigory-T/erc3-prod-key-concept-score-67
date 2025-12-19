from typing import Annotated, Optional, Literal
from annotated_types import Gt, Lt
from pydantic import BaseModel, Field

from .utils import llm_structured, LLM_MODEL_PLAN
from .prompt_preflight import PREFLIGHT_PROMPT


# Denial reasons for preflight check
PreflightDenialReason = Literal[
    "security_violation",
    "vague_or_ambiguous_request",
    "request_not_supported_by_api",
    "possible_security_violation_check_project",
    "may_pass"
]


class PreflightResult(BaseModel):
    """Result of preflight security check."""
    current_actor: str = Field(..., description="Identified current actor from context")
    explanation: Optional[str] = Field(None, description="Brief explanation of the preflight check decision")
    denial_reason: PreflightDenialReason = Field(..., description="Categorization of the request")
    confidence: Annotated[int, Gt(0), Lt(6)] = Field(..., description="Confidence level 1-5 (5 = very confident)")


class PreflightDecision(BaseModel):
    """Final preflight decision after evaluation."""
    should_proceed: bool
    denial_outcome: Optional[str] = None  # 'denied_security' or 'none_unsupported'
    denial_message: Optional[str] = None
    explanation: Optional[str] = None


def _format_user_context(user_context: dict) -> str:
    """Format user context for display in prompts."""
    if not user_context:
        user_context = {}
    
    def fmt(value):
        if value is None or value == "" or value == []:
            return "None (empty)"
        return str(value)
    
    is_public = user_context.get('is_public', False)
    public_warning = " # CRITICAL: This is a GUEST/UNAUTHENTICATED user!" if is_public else ""
    
    return f"""- current_user: {fmt(user_context.get('current_user'))}
- is_public: {fmt(user_context.get('is_public'))}{public_warning}
- department: {fmt(user_context.get('department'))}
- location: {fmt(user_context.get('location'))}
- today: {fmt(user_context.get('today'))}"""


def preflight_check(task: str, user_context: dict = None, wiki_rules: str = "") -> PreflightDecision:
    """
    Perform preflight security check on a task before planning.
    
    This is a fast, lightweight check to catch obvious security violations
    and unsupported requests before investing in full planning.
    
    Args:
        task: The task text to check
        user_context: User context from who_am_i()
        wiki_rules: Distilled wiki rules for context
        
    Returns:
        PreflightDecision with should_proceed=True if planning can continue,
        or should_proceed=False with denial details if request should be denied.
    """
    user_context_str = _format_user_context(user_context)
    prompt = PREFLIGHT_PROMPT.format(
        task=task, 
        user_context=user_context_str,
        wiki_rules=wiki_rules or "# Wiki Rules: Not available"
    )
    
    # Get preflight assessment
    result = llm_structured(prompt, PreflightResult, model=LLM_MODEL_PLAN)
    
    # High confidence denials are immediately rejected
    if result.confidence >= 4:
        if result.denial_reason == "security_violation":
            return PreflightDecision(
                should_proceed=False,
                denial_outcome="denied_security",
                denial_message="Security check failed",
                explanation=result.explanation
            )
        
        if result.denial_reason == "vague_or_ambiguous_request":
            return PreflightDecision(
                should_proceed=False,
                denial_outcome="none_unsupported",
                denial_message="Task is too vague or requires subjective criteria",
                explanation=result.explanation
            )
        
        if result.denial_reason == "request_not_supported_by_api":
            return PreflightDecision(
                should_proceed=False,
                denial_outcome="none_unsupported",
                denial_message="Requested functionality is not supported",
                explanation=result.explanation
            )
    
    # All other cases proceed to planning
    return PreflightDecision(
        should_proceed=True,
        explanation=result.explanation
    )
