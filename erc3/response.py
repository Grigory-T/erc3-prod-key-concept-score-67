import os
import time
import json
from typing import List, Literal, Optional, TYPE_CHECKING
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

from .utils import LLM_MODEL_RESPONSE, _log_llm_call
from .prompt_response import RESPONSE_DECISION_PROMPT

if TYPE_CHECKING:
    from .logger import TaskLogger

load_dotenv()


# ============================================================
# Response Models
# ============================================================

class AgentLink(BaseModel):
    kind: Literal['employee', 'customer', 'project', 'wiki', 'location']
    id: str


class OutcomeCheck(BaseModel):
    reasoning: str = Field(..., description="Why this outcome applies or doesn't apply")
    applies: bool = Field(..., description="True if this outcome applies to the situation")


class ResponseClassification(BaseModel):
    company_policies: str = Field(..., description="For PUBLIC/GUEST users: Check if any rule requires mentioning specific text (like company names, ownership) in responses. If such rule exists, it applies to ALL public responses regardless of topic. State: 'MUST APPLY: [rule]' or 'No mention rules for this user type'.")

    message: str = Field(..., description="Response message with the actual answer. CRITICAL: If company_policies says 'MUST APPLY', you MUST include the required mention in this message together with the answer.")
    
    # Clear logical names for the LLM (mapped to ERC3 codes internally)
    permission_denied_critical: OutcomeCheck = Field(..., description="Critical actions like data deletion - always denied")
    functionality_not_available: OutcomeCheck = Field(..., description="No such functionality in the system")
    not_sufficient_rights: OutcomeCheck = Field(..., description="User lacks permission to execute")
    more_info_needed: OutcomeCheck = Field(..., description="Need more info from user to proceed")
    unclear_task: OutcomeCheck = Field(..., description="Task is vague/unclear or requires subjective judgment")
    system_error: OutcomeCheck = Field(..., description="Technical error occurred during execution")
    object_not_found: OutcomeCheck = Field(..., description="Requested object doesn't exist in system")
    task_completed: OutcomeCheck = Field(..., description="Task completed successfully")
    
    links: List[AgentLink] = Field(default_factory=list, description="Related entity links (only for success outcomes)")


class ResponseDecision(BaseModel):
    message: str
    outcome: Literal[
        'ok_answer',
        'ok_not_found', 
        'denied_security',
        'none_clarification_needed',
        'none_unsupported',
        'error_internal'
    ]
    links: List[AgentLink] = Field(default_factory=list)


# ============================================================
# OpenRouter Client for GPT-4o
# ============================================================

def get_openrouter_client() -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )


def classify_response(prompt: str) -> ResponseClassification:
    client = get_openrouter_client()
    
    # Use structured output with JSON schema
    # max_tokens must be high enough to complete the JSON response
    started = time.time()
    response = client.beta.chat.completions.parse(
        model=LLM_MODEL_RESPONSE,
        messages=[{"role": "user", "content": prompt}],
        response_format=ResponseClassification,
        max_tokens=4096,  # Ensure enough space for full JSON response
    )
    choice = response.choices[0].message
    completion_text = ""
    if hasattr(choice, "content") and choice.content:
        completion_text = choice.content if isinstance(choice.content, str) else json.dumps(choice.content)
    elif hasattr(choice, "parsed") and choice.parsed:
        try:
            completion_text = choice.parsed.model_dump_json()
        except Exception:
            completion_text = str(choice.parsed)
    _log_llm_call(LLM_MODEL_RESPONSE, time.time() - started, response.usage, completion_text)
    
    return response.choices[0].message.parsed


# ============================================================
# Priority-based Outcome Selection
# ============================================================

def select_outcome(classification: ResponseClassification) -> str:
    """
    Select the final outcome based on priority order.
    Maps clear field names to ERC3 outcome codes.
    
    Priority (highest to lowest):
    1. permission_denied_critical → 'denied_security' (data deletion etc)
    2. functionality_not_available → 'none_unsupported'
    3. not_sufficient_rights → 'denied_security'
    4. more_info_needed → 'none_clarification_needed'
    5. unclear_task → 'none_clarification_needed' (same as more_info_needed)
    6. system_error → 'error_internal'
    7. object_not_found → 'ok_not_found'
    8. task_completed → 'ok_answer' (default)
    """
    # Check in priority order, map to ERC3 codes
    if classification.permission_denied_critical.applies:
        return 'denied_security'
    
    if classification.functionality_not_available.applies:
        return 'none_unsupported'
    
    if classification.not_sufficient_rights.applies:
        return 'denied_security'
    
    if classification.more_info_needed.applies:
        return 'none_clarification_needed'
    
    if classification.unclear_task.applies:
        return 'none_clarification_needed'
    
    if classification.system_error.applies:
        return 'error_internal'
    
    if classification.object_not_found.applies:
        return 'ok_not_found'
    
    # Default to ok_answer
    return 'ok_answer'


# ============================================================
# Classification to Log Format
# ============================================================

def build_classification_checks(classification: ResponseClassification) -> dict:
    return {
        "permission_denied_critical": {
            "applies": classification.permission_denied_critical.applies,
            "reasoning": classification.permission_denied_critical.reasoning,
            "erc_code": "denied_security"
        },
        "functionality_not_available": {
            "applies": classification.functionality_not_available.applies,
            "reasoning": classification.functionality_not_available.reasoning,
            "erc_code": "none_unsupported"
        },
        "not_sufficient_rights": {
            "applies": classification.not_sufficient_rights.applies,
            "reasoning": classification.not_sufficient_rights.reasoning,
            "erc_code": "denied_security"
        },
        "more_info_needed": {
            "applies": classification.more_info_needed.applies,
            "reasoning": classification.more_info_needed.reasoning,
            "erc_code": "none_clarification_needed"
        },
        "unclear_task": {
            "applies": classification.unclear_task.applies,
            "reasoning": classification.unclear_task.reasoning,
            "erc_code": "none_clarification_needed"
        },
        "system_error": {
            "applies": classification.system_error.applies,
            "reasoning": classification.system_error.reasoning,
            "erc_code": "error_internal"
        },
        "object_not_found": {
            "applies": classification.object_not_found.applies,
            "reasoning": classification.object_not_found.reasoning,
            "erc_code": "ok_not_found"
        },
        "task_completed": {
            "applies": classification.task_completed.applies,
            "reasoning": classification.task_completed.reasoning,
            "erc_code": "ok_answer"
        }
    }


def decide_response(
    task: str,
    agent_answer: str,
    user_context: dict = None,
    wiki_rules: str = "",
    logger: Optional["TaskLogger"] = None
) -> ResponseDecision:
    # Format user context
    user_str = "Unknown"
    if user_context:
        is_public = user_context.get('is_public', False)
        user_str = f"employee_id: {user_context.get('current_user')}, is_public: {is_public} {'(GUEST/UNAUTHENTICATED!)' if is_public else ''}, department: {user_context.get('department')}, location: {user_context.get('location')}"
    
    # Build prompt
    prompt = RESPONSE_DECISION_PROMPT.format(
        task=task,
        agent_answer=agent_answer,
        user=user_str,
        wiki_rules=wiki_rules or "# Wiki Rules: Not available"
    )
    
    # Get classification from GPT-4o
    classification = classify_response(prompt)
    
    # Select outcome based on priority
    outcome = select_outcome(classification)
    
    # Include links for ALL outcomes (entities identified during execution)
    links = classification.links if classification.links else []
    
    # Log classification if logger provided
    if logger:
        classification_checks = build_classification_checks(classification)
        logger.log_response_classification(
            agent_answer=agent_answer,
            classification_checks=classification_checks,
            selected_outcome=outcome,
            final_message=classification.message,
            links=[{"kind": l.kind, "id": l.id} for l in links] if links else None,
            full_prompt=prompt,
            task=task,
            user_context=user_str,
            wiki_rules=wiki_rules,
            company_policies_check=classification.company_policies
        )
    
    return ResponseDecision(
        message=classification.message,
        outcome=outcome,
        links=links
    )
