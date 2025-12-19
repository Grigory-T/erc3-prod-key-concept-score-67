import os
import re
import json
import time
import requests
from dataclasses import dataclass
from typing import Optional, List, Any
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()


# ============================================================
# Data Classes (mirrors executor/dev_functions.py)
# These are used to reconstruct Pydantic models from JSON dicts
# returned by the Docker executor
# ============================================================

class SkillLevel(BaseModel):
    """Skill or will level."""
    name: str
    level: int


class EmployeeBrief(BaseModel):
    """Brief employee info from list."""
    id: str
    name: str
    email: str
    salary: int
    location: str
    department: str


class EmployeeFull(BaseModel):
    """Full employee info."""
    id: str
    name: str
    email: str
    salary: int
    notes: str
    location: str
    department: str
    skills: List[SkillLevel] = []
    wills: List[SkillLevel] = []


class CustomerBrief(BaseModel):
    """Brief customer info from list."""
    id: str
    name: str
    location: str
    deal_phase: str
    high_level_status: str


class CustomerFull(BaseModel):
    """Full customer info."""
    id: str
    name: str
    brief: str
    location: str
    deal_phase: str
    high_level_status: str
    account_manager: str
    primary_contact_name: Optional[str] = None
    primary_contact_email: Optional[str] = None


class ProjectBrief(BaseModel):
    """Brief project info from list."""
    id: str
    name: str
    customer: str
    status: str


class TeamMember(BaseModel):
    """Project team member."""
    employee: str
    time_slice: float
    role: str


class ProjectFull(BaseModel):
    """Full project info."""
    id: str
    name: str
    description: str
    customer: str
    status: str
    team: List[TeamMember] = []


class TimeEntry(BaseModel):
    """Time entry info."""
    id: str
    employee: str
    customer: Optional[str] = None
    project: Optional[str] = None
    date: str
    hours: float
    work_category: str
    notes: str
    billable: bool
    status: str


class WhoAmI(BaseModel):
    """Current user info."""
    current_user: Optional[str] = None
    is_public: bool = False
    location: Optional[str] = None
    department: Optional[str] = None
    today: Optional[str] = None
    wiki_sha1: Optional[str] = None


# Model registry for reconstruction
_MODEL_SIGNATURES = {
    # EmployeeBrief: id, name, email, salary, location, department (6 fields, no notes)
    frozenset(['id', 'name', 'email', 'salary', 'location', 'department']): EmployeeBrief,
    # EmployeeFull: has notes, skills, wills
    frozenset(['id', 'name', 'email', 'salary', 'notes', 'location', 'department', 'skills', 'wills']): EmployeeFull,
    # CustomerBrief: id, name, location, deal_phase, high_level_status
    frozenset(['id', 'name', 'location', 'deal_phase', 'high_level_status']): CustomerBrief,
    # CustomerFull: has brief, account_manager, primary_contact_*
    frozenset(['id', 'name', 'brief', 'location', 'deal_phase', 'high_level_status', 'account_manager']): CustomerFull,
    # ProjectBrief: id, name, customer, status
    frozenset(['id', 'name', 'customer', 'status']): ProjectBrief,
    # ProjectFull: has description, team
    frozenset(['id', 'name', 'description', 'customer', 'status', 'team']): ProjectFull,
    # TeamMember: employee, time_slice, role
    frozenset(['employee', 'time_slice', 'role']): TeamMember,
    # TimeEntry: id, employee, date, hours, work_category, notes, billable, status
    frozenset(['id', 'employee', 'date', 'hours', 'work_category', 'notes', 'billable', 'status']): TimeEntry,
    # WhoAmI: current_user, is_public, location, department, today, wiki_sha1
    frozenset(['current_user', 'is_public']): WhoAmI,
    # SkillLevel: name, level
    frozenset(['name', 'level']): SkillLevel,
}


def _reconstruct_model(data: Any) -> Any:
    """
    Recursively reconstruct Pydantic models from dicts.
    Returns the original data if not a recognized model dict.
    """
    if isinstance(data, dict):
        # Try to match dict keys to a known model
        keys = frozenset(data.keys())
        
        # Check for exact or subset match
        for sig_keys, model_class in _MODEL_SIGNATURES.items():
            if sig_keys.issubset(keys):
                try:
                    # Recursively reconstruct nested dicts first
                    reconstructed_data = {}
                    for k, v in data.items():
                        reconstructed_data[k] = _reconstruct_model(v)
                    return model_class.model_validate(reconstructed_data)
                except Exception:
                    pass  # Fall through if validation fails
        
        # No match found, recursively process dict values
        return {k: _reconstruct_model(v) for k, v in data.items()}
    
    elif isinstance(data, list):
        return [_reconstruct_model(item) for item in data]
    
    elif isinstance(data, tuple):
        return tuple(_reconstruct_model(item) for item in data)
    
    return data


# API-level stop sequences (passed to API directly)
API_STOP_SEQUENCES = ["Observation:", "Calling tools:", "</json>"]

# Post-processing stop sequences (cut response after these)
POST_STOP_SEQUENCES = [
    "```\n",  # Stop after code block closes
]


def _is_inside_code_block(text: str, position: int) -> bool:
    """
    Check if position is inside a code block (between ``` markers).
    
    Logic: Count triple backticks before the position.
    - Even count (0, 2, 4...) = NOT inside code block
    - Odd count (1, 3, 5...) = INSIDE code block
    """
    prefix = text[:position]
    count = prefix.count("```")
    return count % 2 == 1


def _find_valid_stop_position(output: str, seq: str) -> int:
    """
    Find position of stop sequence, with special handling for </json>.
    
    For </json>: skip occurrences inside code blocks.
    For other sequences: return first occurrence.
    """
    if seq != "</json>":
        return output.find(seq)
    
    # For </json>, find first occurrence NOT inside a code block
    search_start = 0
    while True:
        pos = output.find(seq, search_start)
        if pos == -1:
            return -1  # Not found
        if not _is_inside_code_block(output, pos):
            return pos  # Valid position (not inside code)
        # Skip this occurrence, continue searching
        search_start = pos + len(seq)

# ============================================================
# Model Configuration
# ============================================================

# Provider selection: "openrouter" or "cerebras"
LLM_PROVIDER = "openrouter"

# OpenRouter models (used when LLM_PROVIDER = "openrouter")
# Options: "deepseek/deepseek-v3.2" (reasoning), "deepseek/deepseek-chat-v3-0324" (no reasoning)
LLM_MODEL_AGENT = "deepseek/deepseek-v3.2"  # Main agent model - supports reasoning
LLM_MODEL_DEFAULT = LLM_MODEL_AGENT
LLM_MODEL_PLAN = "openai/gpt-5.1"
LLM_MODEL_DECISION = "openai/gpt-4.1"
LLM_MODEL_REPLAN = "openai/gpt-4.1"

# Cerebras model (used when LLM_PROVIDER = "cerebras")
LLM_MODEL_CEREBRAS = "qwen-3-235b-a22b-instruct-2507"

# Response classification (ALWAYS OpenRouter - needs structured output)
LLM_MODEL_RESPONSE = "openai/gpt-4.1"

# ============================================================
# Reasoning Configuration (for agent step logic ONLY)
# ============================================================
# Set to None to disable reasoning, or one of: "high", "medium", "low", "minimal"
REASONING_EFFORT = "low"  # None = disabled, "high"/"medium"/"low"/"minimal" = enabled

# Model options:
# OpenRouter: "deepseek/deepseek-v3.2" (reasoning), "deepseek/deepseek-chat-v3-0324" (no reasoning), "openai/gpt-4o", "openai/gpt-4.1"
# Cerebras: "llama-3.3-70b" 
# openai/gpt-5.1

# ============================================================
# LLM Logging Callback (set from runner.py)
# ============================================================

_llm_logger = None  # Callable injected from runner
_current_task_id = None  # Set per-task from runner

def set_llm_logger(logger_func, task_id: str = None):
    """Set the LLM logging callback and current task_id."""
    global _llm_logger, _current_task_id
    _llm_logger = logger_func
    _current_task_id = task_id

def _log_llm_call(model: str, duration_sec: float, usage, completion: str):
    """Internal: log LLM call if logger is configured (new typed telemetry)."""
    if not (_llm_logger and _current_task_id):
        return
    
    completion_text = (completion or "").strip()
    
    def _safe_int(val) -> int:
        try:
            return int(val) if val is not None else 0
        except Exception:
            return 0
    
    prompt_tokens = 0
    completion_tokens = 0
    cached_prompt_tokens = 0
    
    if usage:
        # Handle both dict-like and object usage payloads
        prompt_tokens = _safe_int(getattr(usage, "prompt_tokens", None) or getattr(getattr(usage, "input_tokens", None), "total_tokens", None))
        completion_tokens = _safe_int(getattr(usage, "completion_tokens", None) or getattr(getattr(usage, "output_tokens", None), "total_tokens", None))
        
        details = getattr(usage, "prompt_tokens_details", None) or getattr(usage, "input_tokens", None) or {}
        if isinstance(details, dict):
            cached_prompt_tokens = _safe_int(details.get("cached_tokens"))
        else:
            cached_prompt_tokens = _safe_int(getattr(details, "cached_tokens", None))
    
    try:
        # Preferred path: new SDK signature with typed fields + completion text
        _llm_logger(
            task_id=_current_task_id,
            model=model,
            completion=completion_text,
            duration_sec=duration_sec,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
        )
        return
    except TypeError:
        # Fallback to legacy SDK signature (usage BaseModel)
        try:
            class LegacyUsage(BaseModel):
                prompt_tokens: int = 0
                completion_tokens: int = 0
                prompt_tokens_details: dict = {}
            legacy_usage = LegacyUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                prompt_tokens_details={"cached_tokens": cached_prompt_tokens},
            )
            _llm_logger(task_id=_current_task_id, model=model, duration_sec=duration_sec, usage=legacy_usage)
        except Exception:
            pass  # Don't fail on logging errors
    except Exception:
        pass  # Don't fail on logging errors

# ============================================================
# JSON Extraction
# ============================================================

def extract_json(text: str) -> str:
    """Extract JSON from text, handling <json> tags and markdown code blocks."""
    patterns = [
        r'<json>\s*([\s\S]*?)\s*</json>',  # Preferred: <json> tags
        r'```json\s*([\s\S]*?)\s*```',      # Fallback: markdown json
        r'```\s*([\s\S]*?)\s*```',          # Fallback: any code block
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return text.strip()


# ============================================================
# Structured LLM Call
# ============================================================

MAX_JSON_RETRIES = 1  # Number of retry attempts for JSON parsing failures

def llm_structured(prompt: str, response_model: type[BaseModel], model: str = None) -> BaseModel:
    """
    Call LLM and get structured response.
    
    Uses manual JSON extraction with retry logic for empty/malformed responses.
    
    Args:
        prompt: The prompt text
        response_model: Pydantic model class for response
        model: Optional model override (defaults to LLM_MODEL_DEFAULT)
        
    Returns:
        Parsed Pydantic model instance
    """
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )
    
    use_model = model or LLM_MODEL_DEFAULT
    schema = response_model.model_json_schema()
    full_prompt = f"""{prompt}

REQUIRED OUTPUT FORMAT:
Return a JSON object with actual values (NOT the schema itself). Example structure:
{json.dumps({k: f"<your {k}>" for k in schema.get("required", schema.get("properties", {}).keys())}, indent=2)}

Schema reference (fill in actual values):
{json.dumps(schema, indent=2)}

Return ONLY raw JSON, no markdown, no explanation."""
    
    last_error = None
    for attempt in range(MAX_JSON_RETRIES + 1):
        try:
            started = time.time()
            response = client.chat.completions.create(
                model=use_model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0,
                max_tokens=16000
            )
            completion_text = response.choices[0].message.content
            _log_llm_call(use_model, time.time() - started, response.usage, completion_text)
            
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError(f"Empty response from LLM (attempt {attempt + 1})")
            
            json_str = extract_json(content)
            if not json_str or not json_str.strip():
                raise ValueError(f"Empty JSON after extraction (attempt {attempt + 1})")
            
            # Check for empty object {} which DeepSeek sometimes returns
            stripped = json_str.strip()
            if stripped == '{}' or stripped == '[]':
                raise ValueError(f"Empty JSON object/array returned (attempt {attempt + 1})")
            
            return response_model.model_validate_json(json_str)
            
        except Exception as e:
            last_error = e
            if attempt < MAX_JSON_RETRIES:
                time.sleep(1)  # Brief pause before retry
                continue
            else:
                # Final attempt failed - raise the error
                raise last_error


def llm(messages: list, task_id: str = None, model: str = None) -> str:
    """
    Call LLM with messages.
    
    Args:
        messages: List of message dicts with role/content
        task_id: Optional task ID (for context, logging handled elsewhere)
        model: Optional model override (defaults to LLM_MODEL_DEFAULT)
        
    Returns:
        LLM response text
    """
    started = time.time()
    
    if LLM_PROVIDER == "cerebras":
        from cerebras.cloud.sdk import Cerebras
        client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
        use_model = LLM_MODEL_CEREBRAS
        response = client.chat.completions.create(
            model=use_model,
            messages=messages,
            temperature=0,
            max_tokens=10_000,
        )
    else:  # openrouter (default)
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        use_model = model or LLM_MODEL_DEFAULT
        response = client.chat.completions.create(
            model=use_model,
            messages=messages,
            temperature=0,
            max_tokens=10_000,
            stop=API_STOP_SEQUENCES
        )
    
    completion_text = response.choices[0].message.content
    _log_llm_call(use_model, time.time() - started, response.usage, completion_text)
    
    output = response.choices[0].message.content.strip()
    
    # Check if stopped at </json> - API removes stop sequence, add it back
    finish_reason = response.choices[0].finish_reason
    if finish_reason == "stop" and "<json>" in output and "</json>" not in output:
        output = output + "</json>"
    
    # Also check if </final_answer> was cut (e.g., came after </json>)
    if "<final_answer>" in output and "</final_answer>" not in output:
        output = output + "</final_answer>"
    
    # Post-processing: find earliest stop sequence and cut there
    first_stop = None
    first_pos = len(output)
    for seq in POST_STOP_SEQUENCES:
        pos = _find_valid_stop_position(output, seq)
        if pos != -1 and pos < first_pos:
            first_pos = pos
            first_stop = seq
    
    if first_stop:
        output = output[:first_pos + len(first_stop)]
    
    return output


@dataclass
class LLMResponseWithReasoning:
    """LLM response with optional reasoning."""
    content: str  # Main response content (for messages)
    reasoning: Optional[str] = None  # Reasoning/thinking text (for logs only)


def llm_with_reasoning(messages: list, task_id: str = None, model: str = None, enable_reasoning: bool = True) -> LLMResponseWithReasoning:
    """
    Call LLM with optional reasoning support.
    
    For models that support reasoning (like deepseek-v3.2), this returns both
    the main content and the reasoning text separately.
    
    Args:
        messages: List of message dicts with role/content
        task_id: Optional task ID (for context, logging handled elsewhere)
        model: Optional model override (defaults to LLM_MODEL_DEFAULT)
        enable_reasoning: Whether to request reasoning (default True, uses REASONING_EFFORT)
        
    Returns:
        LLMResponseWithReasoning with content and optional reasoning
    """
    started = time.time()
    
    if LLM_PROVIDER == "cerebras":
        # Cerebras doesn't support reasoning
        from cerebras.cloud.sdk import Cerebras
        client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
        use_model = LLM_MODEL_CEREBRAS
        response = client.chat.completions.create(
            model=use_model,
            messages=messages,
            temperature=0,
            max_tokens=10_000,
        )
        content = response.choices[0].message.content or ""
        _log_llm_call(use_model, time.time() - started, response.usage, content)
        return LLMResponseWithReasoning(content=_post_process_output(content))
    
    # OpenRouter with optional reasoning
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )
    use_model = model or LLM_MODEL_DEFAULT
    
    # Build request kwargs
    kwargs = {
        "model": use_model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 10_000,
        "stop": API_STOP_SEQUENCES,
    }
    
    # Add reasoning config if enabled and configured
    if enable_reasoning and REASONING_EFFORT:
        kwargs["extra_body"] = {
            "reasoning": {
                "effort": REASONING_EFFORT,
                "exclude": False  # Include reasoning in response
            }
        }
    
    response = client.chat.completions.create(**kwargs)
    
    msg = response.choices[0].message
    content = msg.content or ""
    
    # Extract reasoning from response
    reasoning_text = None
    
    # Try to get reasoning from various fields
    # 1. Direct reasoning field (OpenRouter normalized)
    if hasattr(msg, 'reasoning') and msg.reasoning:
        reasoning_text = msg.reasoning
    
    # 2. reasoning_details array (structured format)
    elif hasattr(msg, 'reasoning_details') and msg.reasoning_details:
        reasoning_parts = []
        for detail in msg.reasoning_details:
            if hasattr(detail, 'text') and detail.text:
                reasoning_parts.append(detail.text)
            elif hasattr(detail, 'summary') and detail.summary:
                reasoning_parts.append(f"[Summary] {detail.summary}")
            elif isinstance(detail, dict):
                if detail.get('text'):
                    reasoning_parts.append(detail['text'])
                elif detail.get('summary'):
                    reasoning_parts.append(f"[Summary] {detail['summary']}")
        if reasoning_parts:
            reasoning_text = "\n".join(reasoning_parts)
    
    # Log the LLM call (content only, reasoning handled separately in agent)
    _log_llm_call(use_model, time.time() - started, response.usage, content)
    
    # Post-process the output
    output = _post_process_output(content)
    
    return LLMResponseWithReasoning(content=output, reasoning=reasoning_text)


def _post_process_output(content: str) -> str:
    """Post-process LLM output: handle stop sequences, fix tags."""
    output = content.strip()
    
    # Check if stopped at </json> - API removes stop sequence, add it back
    if "<json>" in output and "</json>" not in output:
        output = output + "</json>"
    
    # Also check if </final_answer> was cut (e.g., came after </json>)
    if "<final_answer>" in output and "</final_answer>" not in output:
        output = output + "</final_answer>"
    
    # Post-processing: find earliest stop sequence and cut there
    first_stop = None
    first_pos = len(output)
    for seq in POST_STOP_SEQUENCES:
        pos = _find_valid_stop_position(output, seq)
        if pos != -1 and pos < first_pos:
            first_pos = pos
            first_stop = seq
    
    if first_stop:
        output = output[:first_pos + len(first_stop)]
    
    return output


def parse_code(response: str) -> str:
    """Extract code from ```python blocks."""
    pattern = r"```python\s*(.*?)\s*```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return "\n\n".join(match.strip() for match in matches)
    return None


def check_response_format(response: str) -> tuple[bool, bool, str]:
    """
    Check if response has required format: <final_answer> AND <json>.
    
    Returns:
        (has_final_answer, has_json, error_message)
        error_message is None if both are present
    """
    # Check for COMPLETE tag pairs (opening AND closing)
    has_final_open = "<final_answer>" in response
    has_final_close = "</final_answer>" in response
    has_final = has_final_open and has_final_close
    
    has_json = bool(re.search(r"<json>.*?</json>", response, re.DOTALL))
    
    if has_final and has_json:
        return True, True, None
    elif has_final_open and not has_final_close:
        return True, False, "Missing </final_answer> closing tag."
    elif has_final and not has_json:
        return True, False, "Missing <json> block. Add <json>your JSON</json> after </final_answer>."
    elif not has_final and has_json:
        return False, True, "Missing <final_answer> block. Add <final_answer>your conclusion</final_answer> before <json>."
    elif has_final_open:
        # Has opening but no closing, and no json
        return True, False, "Missing </final_answer> closing tag and <json> block."
    else:
        return False, False, None  # No answer attempted


def parse_final_answer(response: str) -> str:
    """
    Extract text from <final_answer> tags AND <json> blocks.
    Returns combined answer ONLY if BOTH are present.
    Returns None if either is missing.
    """
    # Check both tags are present
    has_final, has_json, _ = check_response_format(response)
    if not (has_final and has_json):
        return None
    
    parts = []
    
    # 1. Extract <final_answer> content
    pattern_closed = r"<final_answer>(.*?)</final_answer>"
    matches = re.findall(pattern_closed, response, re.DOTALL)
    if matches:
        for match in matches:
            # Clean: remove any <json> blocks from inside final_answer
            cleaned = re.sub(r"<json>\s*.*?\s*</json>", "", match, flags=re.DOTALL).strip()
            if cleaned:
                parts.append(cleaned)
    
    # 2. Extract <json> content
    json_pattern = r"<json>\s*(.*?)\s*</json>"
    json_matches = re.findall(json_pattern, response, re.DOTALL)
    if json_matches:
        parts.extend(json_matches)
    
    if parts:
        return "\n\n".join(parts)
    return None


def format_code_result(result: dict) -> str:
    """Format code execution result with real newlines."""
    parts = []
    if result.get("stdout"):
        parts.append(f"STDOUT:\n{result['stdout']}")
    if result.get("error"):
        parts.append(f"ERROR:\n{result['error']}")
    if result.get("result") is not None:
        parts.append(f"RESULT:\n{result['result']}")
    return "\n".join(parts) if parts else "(no output)"


def execute_code(code: str, session_id: str, task_id: str) -> dict:
    """
    Execute code in isolated Python environment (Docker).
    
    Args:
        code: Python code to execute
        session_id: Session ID
        task_id: Task ID
        
    Returns:
        Dict with stdout, error, result (result reconstructed as Pydantic models where possible)
    """
    interpreter_id = f"{session_id or 'default'}_{task_id or 'default'}"
    
    payload = {
        "code": code,
        "interpreter_id": interpreter_id,
        "session_id": session_id,
        "task_id": task_id
    }
    response = requests.post(
        "http://localhost:8001/exec",
        json=payload,
        timeout=300  # 5 minutes per code execution
    )
    response.raise_for_status()
    result = response.json()
    
    # Reconstruct Pydantic models from JSON dicts in result
    if result.get("result") is not None:
        result["result"] = _reconstruct_model(result["result"])
    
    return result


def format_messages(messages: list) -> str:
    """Format messages for logging/debugging."""
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        parts.append(f"=== {role.upper()} ===\n{content}")
    return "\n\n".join(parts)
