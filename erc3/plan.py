import json
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from .utils import llm_structured, LLM_MODEL_PLAN, LLM_MODEL_DECISION, LLM_MODEL_REPLAN
from .prompt_plan import PLAN_PROMPT, DECISION_PROMPT, REPLAN_REMAINING_PROMPT

class PlanStep(BaseModel):
    step_number: int = Field(..., description="Step number (1-based)")
    description: str = Field(..., description="What this step should accomplish")
    approach: str = Field(..., description="How to accomplish this step")
    expected_output: str = Field(..., description="Valid JSON Schema for step output. MUST be valid parseable JSON. Example: {\"type\": \"object\", \"properties\": {\"employee_id\": {\"type\": \"string\"}, \"found\": {\"type\": \"boolean\"}}, \"required\": [\"employee_id\", \"found\"]}")
    
    @field_validator('expected_output')
    @classmethod
    def validate_expected_output_is_json_schema(cls, v: str) -> str:
        """Ensure expected_output is valid JSON Schema."""
        try:
            parsed = json.loads(v)
        except json.JSONDecodeError as e:
            raise ValueError(f"expected_output must be valid JSON: {e}")
        
        # Check it looks like a JSON Schema (has "type" or "properties")
        if isinstance(parsed, dict):
            if "type" not in parsed and "properties" not in parsed:
                raise ValueError(f"expected_output must be JSON Schema with 'type' or 'properties', got: {v[:100]}")
        return v


class ExecutionPlan(BaseModel):
    immediately_abort: bool = Field(False, description="Set to True to abort task immediately (vague task, security violation, etc.)")
    abort_reason: Optional[str] = Field(None, description="Reason for aborting (required if immediately_abort=True)")
    steps: List[PlanStep] = Field(default_factory=list, description="List of steps to execute (5-15 steps, empty if immediately_abort=True)")


class StepDecision(BaseModel):
    reasoning: str = Field(..., description="Why this decision was made")

    decision: Literal["continue", "final_answer", "abort", "replan_remaining"] = Field(
        ..., description="What to do next"
    )

    # For FINAL_ANSWER - the answer to return
    final_answer: Optional[str] = Field(None, description="Plain text string with the final answer (NOT a JSON object, just a string)")
    
    # For ABORT - reason why task cannot be done
    abort_reason: Optional[str] = Field(None, description="Plain text string explaining why task cannot be completed (NOT a JSON object)")
    
    # For REPLAN_REMAINING - what we learned that requires replanning
    new_context: Optional[str] = Field(None, description="New information learned that affects planning")


def _format_user_context(user_context: dict) -> str:
    """Format user context for display in prompts."""
    if not user_context:
        user_context = {}
    
    def fmt(value):
        if value is None or value == "" or value == []:
            return "None (empty)"
        return str(value)
    
    return f"""- current_user: {fmt(user_context.get('current_user'))}
- is_public: {fmt(user_context.get('is_public'))}  # CRITICAL: True = guest/unauthenticated user!
- department: {fmt(user_context.get('department'))}
- location: {fmt(user_context.get('location'))}
- today: {fmt(user_context.get('today'))}"""


def create_plan(task: str, user_context: dict = None, wiki_rules: str = "") -> tuple[List[str], ExecutionPlan]:
    user_context_str = _format_user_context(user_context)
    prompt = PLAN_PROMPT.format(task=task, user_context=user_context_str, wiki_rules=wiki_rules or "# Wiki Rules: Not available")
    plan = llm_structured(prompt, ExecutionPlan, model=LLM_MODEL_PLAN)
    
    # If immediate abort, return empty steps
    if plan.immediately_abort:
        return [], plan
    
    # Convert to list of step descriptions (use sequential numbering from 1)
    steps = []
    for i, step in enumerate(plan.steps, start=1):
        # Add fuzzy match instruction to step description
        step.description += ''  # " (names/terms: use fuzzy match and semantic/logical matching)"
        
        steps.append(
            f"Step {i}: {step.description}\n"
            f"Approach: {step.approach}\n"
            f"Expected Output: {step.expected_output}"
        )
    
    return steps, plan


def format_plan_for_log(plan: ExecutionPlan) -> str:
    if plan.immediately_abort:
        return f"IMMEDIATE ABORT\nReason: {plan.abort_reason}"
    
    lines = ["Steps:"]
    for step in plan.steps:
        lines.append(f"  {step.step_number}. {step.description}")
        lines.append(f"     Approach: {step.approach}")
        lines.append(f"     Expected Output: {step.expected_output}")
    
    return "\n".join(lines)


def make_decision(
    task: str,
    plan: ExecutionPlan,
    completed_steps: List[tuple[str, str]],  # (step_description, result)
    remaining_steps: List[str],
    wiki_rules: str = ""
) -> StepDecision:
    # Format completed steps
    completed_str = ""
    for i, (step_desc, result) in enumerate(completed_steps, 1):
        completed_str += f"\n### Step {i}\n{step_desc}\n\n**Result:**\n{result}\n"
    
    if not completed_str:
        completed_str = "(No steps completed yet)"
    
    # Format remaining steps
    remaining_str = "\n".join(remaining_steps) if remaining_steps else "(No remaining steps)"
    
    # Last result
    last_result = completed_steps[-1][1] if completed_steps else "(No result yet)"
    
    prompt = DECISION_PROMPT.format(
        task=task,
        plan_summary=format_plan_for_log(plan),
        completed_steps=completed_str,
        last_result=last_result,
        remaining_steps=remaining_str,
        wiki_rules=wiki_rules or "# Wiki Rules: Not available"
    )
    
    return llm_structured(prompt, StepDecision, model=LLM_MODEL_DECISION)


def replan_remaining(
    task: str,
    completed_steps: List[tuple[str, str]],
    new_context: str,
    old_remaining_steps: List[str],
    wiki_rules: str = ""
) -> tuple[List[str], ExecutionPlan]:
    # Format completed steps
    completed_str = ""
    for i, (step_desc, result) in enumerate(completed_steps, 1):
        completed_str += f"\n### Step {i}\n{step_desc}\n\n**Result:**\n{result}\n"
    
    next_step_num = len(completed_steps) + 1
    
    prompt = REPLAN_REMAINING_PROMPT.format(
        task=task,
        completed_steps=completed_str or "(None)",
        new_context=new_context or "(None specified)",
        old_remaining_steps="\n".join(old_remaining_steps) if old_remaining_steps else "(None)",
        next_step_number=next_step_num,
        wiki_rules=wiki_rules or "# Wiki Rules: Not available"
    )
    
    plan = llm_structured(prompt, ExecutionPlan, model=LLM_MODEL_REPLAN)
    
    # Convert to list of step descriptions (continue numbering from next_step_num)
    steps = []
    for i, step in enumerate(plan.steps, start=next_step_num):
        steps.append(f"Step {i}: {step.description}\nApproach: {step.approach}")
    
    return steps, plan
