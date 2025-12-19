import re
import json
from dataclasses import dataclass, field
from typing import Optional, Any
from .utils import llm, llm_with_reasoning, parse_code, parse_final_answer, check_response_format, format_code_result, execute_code, LLM_MODEL_AGENT, REASONING_EFFORT
from .plan import create_plan, make_decision, replan_remaining, format_plan_for_log
from .preflight import preflight_check
from .logger import TaskLogger
from .prompt_agent import LLM_LOOP_SYSTEM_PROMPT
from .wiki_rules import get_empty_wiki_rules, create_wiki_rules_context


def validate_json_response(response: str, expected_schema: str = None) -> tuple[bool, str]:
    # 1. Check if <json> block exists
    json_pattern = r"<json>\s*(.*?)\s*</json>"
    matches = re.findall(json_pattern, response, re.DOTALL)
    
    if not matches:
        return False, "Missing <json> block. Please provide your answer in <json>...</json> tags."
    
    # 2. Try to parse as valid JSON
    json_str = matches[0]
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}. Please fix the JSON syntax."
    
    # 3. If expected_schema provided, try basic validation
    if expected_schema:
        try:
            schema = json.loads(expected_schema)
            # Basic check: if schema has "properties", check required keys exist
            if isinstance(schema, dict) and "properties" in schema:
                required = schema.get("required", [])
                missing = [k for k in required if k not in parsed]
                if missing:
                    return False, f"JSON missing required keys: {missing}. Expected schema: {expected_schema}"
            elif isinstance(schema, dict) and "type" not in schema:
                # Schema is just a template like {"key": "value"}
                expected_keys = set(schema.keys())
                actual_keys = set(parsed.keys()) if isinstance(parsed, dict) else set()
                missing = expected_keys - actual_keys
                if missing:
                    return False, f"JSON missing expected keys: {list(missing)}. Template: {expected_schema}"
        except json.JSONDecodeError:
            pass  # Schema itself is not valid JSON, skip validation
    
    return True, None


def extract_expected_output(step_description: str) -> str:
    match = re.search(r"Expected Output:\s*(.+?)(?:\n|$)", step_description)
    if match:
        return match.group(1).strip()
    return None

def _format_context_value(value) -> str:
    """Format context value, clearly indicating empty/None fields."""
    if value is None or value == "" or value == []:
        return "None (empty)"
    return str(value)


def build_step_prompt(step_description: str, user_context: dict, task: str = None, completed_steps: list[tuple[str, str]] = None) -> str:
    parts = []
    
    # 1. Current User Context
    parts.extend([
        "## Current User Context (who_am_i) - **as an assistant you acting on behalf of this User**",
        f"current_user: {_format_context_value(user_context.get('current_user'))}",
        f"department: {_format_context_value(user_context.get('department'))}",
        f"location: {_format_context_value(user_context.get('location'))}",
        f"today date: {_format_context_value(user_context.get('today'))}",
        f"is_public: {_format_context_value(user_context.get('is_public'))}",
    ])
    
    # 2. Global Task
    if task:
        parts.extend([
            "",
            "## GLOBAL TASK (FOR REFERENCE ONLY, DO NOT SOLVE THIS TASK. YOU NEED TO SOLVE THE CURRENT STEP ONLY)",
            f"**Task:** {task}",
        ])
    
    # 3. Previous Steps Completed
    if completed_steps:
        parts.append("\n## Previous Steps of Task Plan Completed")
        for i, (desc, result) in enumerate(completed_steps, 1):
            parts.append(f"\n### Step {i}\n{desc}\n**Result:** {result}")
    
    # 4. Current Step (FOCUS)
    parts.extend([
        "",
        "## >>> CURRENT STEP (FOCUS HERE) <<<",
        "This is the current step you need to execute. Focus on completing THIS step.",
        "",
        step_description
    ])
    
    return "\n".join(parts)

MAX_ITERATIONS_PER_STEP = 15


def _check_wiki_refresh(ctx: "ExecutionContext", messages: list) -> tuple["ExecutionContext", list]:
    """
    Check if wiki SHA changed and refresh wiki rules if needed.
    Updates system prompt in messages if wiki changed.
    
    Called before EVERY LLM message to catch wiki updates in real-time.
    The who_am_i() call is lightweight - 99% of the time there's no change,
    but we need to catch that 1% when it does change.
    """
    if not ctx.dev_api:
        return ctx, messages
    
    try:
        from .wiki_rules import create_wiki_rules_context
        
        # Get current wiki SHA from API
        current_who = ctx.dev_api.who_am_i()
        new_sha1 = current_who.wiki_sha1
        old_sha1 = ctx.user_context.get("wiki_sha1", "")
        
        # If wiki changed, refresh rules
        if new_sha1 and new_sha1 != old_sha1:
            print(f"Wiki changed! Refreshing rules (old={old_sha1[:8] if old_sha1 else 'none'}, new={new_sha1[:8]})")
            
            # Store old rules for logging
            old_rules = ctx.wiki_rules
            
            # Update user context with new SHA
            ctx.user_context["wiki_sha1"] = new_sha1
            ctx.user_context["today"] = current_who.today or ctx.user_context.get("today")
            
            # Re-distill rules with new context
            new_wiki_ctx = create_wiki_rules_context(ctx.dev_api, ctx.user_context)
            new_rules = new_wiki_ctx.get_formatted_rules()
            ctx.wiki_rules = new_rules
            
            # Log wiki change with before/after states
            ctx.log.log_wiki_change(old_sha1, new_sha1, old_rules, new_rules)
            
            # Update system prompt in messages (first message)
            if messages and messages[0].get("role") == "system":
                new_system_prompt = LLM_LOOP_SYSTEM_PROMPT.format(wiki_rules=new_rules)
                messages[0]["content"] = new_system_prompt
                
                # Log the wiki refresh in conversation
                ctx.log.append_message("system", "[WIKI RULES REFRESHED - rules may have changed mid-conversation]")
    except Exception as e:
        # Don't fail on wiki refresh errors
        print(f"Warning: Wiki refresh check failed: {e}")
    
    return ctx, messages


def llm_loop(step_description: str, ctx: "ExecutionContext", completed_steps: list[tuple[str, str]] = None) -> str:
    step_num = len(completed_steps or []) + 1
    
    # Format system prompt with wiki rules
    system_prompt = LLM_LOOP_SYSTEM_PROMPT.format(wiki_rules=ctx.wiki_rules or get_empty_wiki_rules())
    
    user_prompt = build_step_prompt(step_description, ctx.user_context, task=ctx.task, completed_steps=completed_steps)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    # Start step logging with immediate message output
    ctx.log.start_step(step_num, step_description)
    ctx.log.init_messages_file(system_prompt=system_prompt)
    ctx.log.append_message("user", user_prompt)
    
    # Track verification state - show verification prompt only ONCE per step
    verification_shown = False
    
    for iteration in range(MAX_ITERATIONS_PER_STEP):
        # Check wiki for updates before EVERY LLM call
        ctx, messages = _check_wiki_refresh(ctx, messages)
        
        # Get LLM response with optional reasoning
        llm_result = llm_with_reasoning(messages, task_id=ctx.task_id, model=LLM_MODEL_AGENT, enable_reasoning=bool(REASONING_EFFORT))
        response = llm_result.content
        
        # Log reasoning separately (to step logs only, NOT to messages)
        if llm_result.reasoning:
            ctx.log.log_reasoning(llm_result.reasoning)
        
        # Add content (without reasoning) to messages for conversation continuity
        messages.append({"role": "assistant", "content": response})
        ctx.log.append_message("assistant", response)
        
        # === Check for code to execute ===
        # Code takes precedence - if agent wrote code, they want to execute it first
        # Even if final_answer is present, execute code first (agent may have included
        # code to verify/compute something before finalizing)
        code = parse_code(response)
        if code:
            result = execute_code(
                code=code,
                session_id=ctx.session_id,
                task_id=ctx.task_id
            )
            formatted = format_code_result(result)
            user_msg = f"Code execution result:\n{formatted}"
            messages.append({"role": "user", "content": user_msg})
            ctx.log.append_message("user", user_msg)
            continue
        
        # === Check for final answer format ===
        has_final, has_json, format_error = check_response_format(response)
        
        # If partial format (one but not both), ask to complete
        if (has_final or has_json) and not (has_final and has_json):
            fix_msg = (
                f"INCOMPLETE FORMAT: {format_error}\n\n"
                f"Your response MUST have BOTH:\n"
                f"1. <final_answer>Your text conclusion</final_answer>\n"
                f"2. <json>{{\"your\": \"valid JSON\"}}</json>\n\n"
                f"Please provide both."
            )
            messages.append({"role": "user", "content": fix_msg})
            ctx.log.append_message("user", fix_msg)
            continue
        
        # If complete format, parse and validate
        final = parse_final_answer(response)
        if not final and (has_final and has_json):
            # Format detected but extraction failed (empty content?)
            fix_msg = (
                "FORMAT ERROR: Tags detected but content is empty or malformed.\n"
                "Please provide non-empty content in both blocks:\n"
                "1. <final_answer>Your actual conclusion text</final_answer>\n"
                "2. <json>{\"actual\": \"data\"}</json>"
            )
            messages.append({"role": "user", "content": fix_msg})
            ctx.log.append_message("user", fix_msg)
            continue
        
        if final:
            # --- JSON VALIDATION: Check JSON is valid and matches schema ---
            expected_output = extract_expected_output(step_description)
            is_valid, error_msg = validate_json_response(response, expected_output)
            
            if not is_valid:
                # JSON validation failed - ask model to fix it
                fix_msg = (
                    f"JSON ERROR: {error_msg}\n\n"
                    f"Please provide your answer again with:\n"
                    f"1. <final_answer>Your text conclusion</final_answer>\n"
                    f"2. <json>valid JSON matching: {expected_output or 'expected schema'}</json>\n\n"
                    f"Try again."
                )
                messages.append({"role": "user", "content": fix_msg})
                ctx.log.append_message("user", fix_msg)
                continue  # Let model try again
            
            # --- VERIFICATION (only once per step) ---
            # If verification was already shown, accept the valid answer directly
            if verification_shown:
                ctx.log.log_step_conclusion(final)
                return final
            
            # First time with valid answer - show verification prompt
            verification_shown = True
            verification_msg = (
                f"VERIFICATION: Before accepting this conclusion, please confirm:\n\n"
                f"STEP TO ACCOMPLISH:\n{step_description}\n\n"
                f"CHECKLIST:\n"
                f"- Did you execute the step instruction?\n"
                f"- Is there any action you skipped or assumed?\n"
                f"- Are all facts verified (not assumed)?\n\n"
                
                f"JSON response MUST contain ONLY values that were explicitly printed/verified from function returns.\n"
                f"- You did not use any make-up data or sensiable defaults. All facts should be derived from function returns.\n"
                f"- If any object is not found - consider using fuzzy matching, partial matching, double checks, etc.\n"
                f"- Consider applying semantic/logical matching, if term is ambiguous or unclear, or it is abbreviation.\n"
                f"- If nesseary, consider getting the object's fields value to understand the object better (project status, employee role, custome location, etc.)\n\n"

                f"update functions are STATE-CHANGING - YOU SHOULD CALL THEM ONLY ONCE!\n\n"
                f'- if you change state of any object - include explicit full information about this in the <final_answer> section.\n\n'
                
                f"If complete, repeat BOTH your <final_answer> AND <json> blocks.\n"
                f"If something was missed, execute the missing action now.\n\n"

                f"If you are sure about results - you do not need to verify again. Just provide the <final_answer> and <json> blocks."
            )
            messages.append({"role": "user", "content": verification_msg})
            ctx.log.append_message("user", verification_msg)
            
            # Check wiki for updates before verification LLM call
            ctx, messages = _check_wiki_refresh(ctx, messages)
            
            # Get verification response with optional reasoning
            verify_result = llm_with_reasoning(messages, task_id=ctx.task_id, model=LLM_MODEL_AGENT, enable_reasoning=bool(REASONING_EFFORT))
            verification_response = verify_result.content
            
            # Log reasoning separately (to step logs only)
            if verify_result.reasoning:
                ctx.log.log_reasoning(verify_result.reasoning)
            
            messages.append({"role": "assistant", "content": verification_response})
            ctx.log.append_message("assistant", verification_response)
            
            # Check verification response format (must have BOTH <final_answer> AND <json>)
            has_final_v, has_json_v, format_error_v = check_response_format(verification_response)
            
            if has_final_v and has_json_v:
                # Full valid response - parse and return
                verified_final = parse_final_answer(verification_response)
                if verified_final:
                    ctx.log.log_step_conclusion(verified_final)
                    return verified_final
                else:
                    # Format looks valid but extraction failed (e.g., unclosed tags)
                    fix_msg = (
                        "FORMAT ERROR: Tags detected but content extraction failed.\n"
                        "Ensure tags are properly closed:\n"
                        "1. <final_answer>Your text</final_answer>\n"
                        "2. <json>{\"your\": \"json\"}</json>\n\n"
                        "Please provide your answer again with properly closed tags."
                    )
                    messages.append({"role": "user", "content": fix_msg})
                    ctx.log.append_message("user", fix_msg)
                    continue
            
            # Incomplete verification response - prompt to complete
            if (has_final_v or has_json_v) and not (has_final_v and has_json_v):
                fix_msg = (
                    f"INCOMPLETE: {format_error_v}\n"
                    f"Please provide BOTH <final_answer> AND <json> blocks."
                )
                messages.append({"role": "user", "content": fix_msg})
                ctx.log.append_message("user", fix_msg)
                continue
            
            # Model realized something was missed - check for code
            # Code takes precedence - if agent wrote code, execute it first
            code_after_verify = parse_code(verification_response)
            if code_after_verify:
                result = execute_code(
                    code=code_after_verify,
                    session_id=ctx.session_id,
                    task_id=ctx.task_id
                )
                formatted = format_code_result(result)
                user_msg = f"Code execution result:\n{formatted}"
                messages.append({"role": "user", "content": user_msg})
                ctx.log.append_message("user", user_msg)
                continue  # Continue the loop with the new code result
            
            # Neither final_answer nor code - prompt to continue
            prompt_msg = "Please continue. Execute Python code or provide <final_answer>."
            messages.append({"role": "user", "content": prompt_msg})
            ctx.log.append_message("user", prompt_msg)
            continue
        
        # No recognized action - prompt to continue
        prompt_msg = "Please continue. Execute Python code or provide <final_answer>."
        messages.append({"role": "user", "content": prompt_msg})
        ctx.log.append_message("user", prompt_msg)
    
    # Max iterations reached
    conclusion = f"Step incomplete after {MAX_ITERATIONS_PER_STEP} iterations. Last response: {response}"
    ctx.log.log_step_conclusion(conclusion)
    return conclusion

@dataclass
class ExecutionContext:
    """Context for agent execution."""
    session_id: str
    task_id: str
    log: TaskLogger
    task: str = None  # The global task being solved
    user_context: dict = None  # who_am_i() result
    wiki_rules: str = ""  # Distilled wiki rules for injection into prompts
    dev_api: Any = None  # ERC3 dev API client for wiki refresh checks


def format_full_execution_summary(completed_steps: list[tuple[str, str]], final_result: str, end_reason: str) -> str:
    """Format complete execution summary."""
    parts = [f"## Execution Summary ({end_reason})\n"]
    
    for i, (step_desc, step_result) in enumerate(completed_steps, 1):
        parts.append(f"### Step {i}\n{step_desc}\n**Result:** {step_result}\n")
    
    parts.append(f"### Final Result\n{final_result}")
    
    return "\n".join(parts)


MAX_REPLANS = 5
MAX_TOTAL_STEPS = 20


@dataclass
class AgentResult:
    """Result from run_agent including updated context."""
    answer: str
    log: TaskLogger
    wiki_rules: str  # Final wiki rules (may have been updated mid-task)
    user_context: dict  # Final user context (may have been updated mid-task)


def run_agent(task: str, session_id: str, task_id: str, task_index: int = 0, run_id: str = None, user_context: dict = None, dev_api=None, wiki_rules: str = "") -> AgentResult:
    # Create logger
    log = TaskLogger(task_index=task_index, task_text=task, run_id=run_id)
    ctx = ExecutionContext(
        session_id=session_id, 
        task_id=task_id, 
        log=log, 
        task=task, 
        user_context=user_context.copy() if user_context else {},
        wiki_rules=wiki_rules or get_empty_wiki_rules(),
        dev_api=dev_api
    )
    
    # Log initial wiki state
    wiki_sha = ctx.user_context.get("wiki_sha1", "unknown")
    log.log_wiki_initial(wiki_sha, ctx.wiki_rules)
    
    # === PREFLIGHT CHECK (before planning) ===
    # Fast security check to catch obvious violations early
    preflight = preflight_check(task, user_context=ctx.user_context, wiki_rules=ctx.wiki_rules)
    log.log_preflight(
        should_proceed=preflight.should_proceed,
        explanation=preflight.explanation,
        denial_outcome=preflight.denial_outcome,
        denial_message=preflight.denial_message
    )
    
    # If preflight denied, return early without planning
    if not preflight.should_proceed:
        abort_answer = f"Request denied at preflight: {preflight.denial_message}"
        log.log_summary(abort_answer)
        return AgentResult(answer=abort_answer, log=log, wiki_rules=ctx.wiki_rules, user_context=ctx.user_context)
    
    # === PLANNING PHASE ===
    # Create initial execution plan
    plan_steps, plan_obj = create_plan(task, user_context=ctx.user_context, wiki_rules=ctx.wiki_rules)
    
    # Log the plan
    plan_text = format_plan_for_log(plan_obj)
    log.log_plan(plan_text)
    
    # Handle immediate abort from planning
    if plan_obj.immediately_abort:
        abort_answer = f"Task aborted at planning stage: {plan_obj.abort_reason}"
        log.log_summary(abort_answer)
        return AgentResult(answer=abort_answer, log=log, wiki_rules=ctx.wiki_rules, user_context=ctx.user_context)
    
    # Tracking state
    completed_steps: list[tuple[str, str]] = []  # (step_description, result)
    replan_count = 0
    total_steps_executed = 0
    step_index = 0
    
    # Main execution loop
    while step_index < len(plan_steps) and total_steps_executed < MAX_TOTAL_STEPS:
        current_step = plan_steps[step_index]
        remaining_steps = plan_steps[step_index + 1:]
        
        # Execute the current step
        result = llm_loop(current_step, ctx, completed_steps=completed_steps)
        completed_steps.append((current_step, result))
        total_steps_executed += 1
        
        # Make decision about what to do next
        decision = make_decision(
            task=task,
            plan=plan_obj,
            completed_steps=completed_steps,
            remaining_steps=remaining_steps,
            wiki_rules=ctx.wiki_rules
        )
        
        # Check if replan is requested but max replans reached - override decision
        if decision.decision == "replan_remaining" and replan_count >= MAX_REPLANS:
            log.log_decision(
                step_num=total_steps_executed,
                decision="skip_replan",
                reasoning=f"Maximum replans ({MAX_REPLANS}) reached. Original decision was {decision.decision}: {decision.reasoning}"
            )
            step_index += 1
            continue
        
        # Log the decision (only if not overridden above)
        log.log_decision(
            step_num=total_steps_executed,
            decision=decision.decision,
            reasoning=decision.reasoning
        )
        
        # Handle decision
        if decision.decision == "continue":
            step_index += 1
            
        elif decision.decision == "final_answer":
            final = decision.final_answer or result
            full_summary = format_full_execution_summary(completed_steps, final, "final_answer")
            log.log_summary(full_summary)
            return AgentResult(answer=full_summary, log=log, wiki_rules=ctx.wiki_rules, user_context=ctx.user_context)
            
        elif decision.decision == "abort":
            abort_msg = f"Task aborted: {decision.abort_reason}"
            full_summary = format_full_execution_summary(completed_steps, abort_msg, "abort")
            log.log_summary(full_summary)
            return AgentResult(answer=full_summary, log=log, wiki_rules=ctx.wiki_rules, user_context=ctx.user_context)
            
        elif decision.decision == "replan_remaining":
            replan_count += 1
            new_steps, new_plan = replan_remaining(
                task=task,
                completed_steps=completed_steps,
                new_context=decision.new_context or "",
                old_remaining_steps=remaining_steps,
                wiki_rules=ctx.wiki_rules
            )
            
            plan_steps = plan_steps[:step_index + 1] + new_steps
            plan_obj = new_plan
            
            log.log_replan(replan_count, "remaining", format_plan_for_log(new_plan))
            step_index += 1
    
    # Loop ended without explicit final answer
    if completed_steps:
        last_result = completed_steps[-1][1]
        full_summary = format_full_execution_summary(completed_steps, last_result, "limit_reached")
        log.log_summary(full_summary)
        return AgentResult(answer=full_summary, log=log, wiki_rules=ctx.wiki_rules, user_context=ctx.user_context)
    else:
        fallback = "Task could not be completed within step limits."
        log.log_summary(fallback)
        return AgentResult(answer=fallback, log=log, wiki_rules=ctx.wiki_rules, user_context=ctx.user_context)
        
