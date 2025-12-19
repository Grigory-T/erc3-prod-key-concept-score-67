"""
File-based logging for agent execution.

Logs are stored in: logs/run_{id}/task_{index}/

Structure:
    task_{index}/
        task.txt           - Original task
        wiki_changes.txt   - Wiki state history (initial + all changes)
        preflight.txt      - Preflight security check result
        plan.txt           - Initial plan + replans
        step_N/
            prompt.txt     - Step description
            messages.txt   - LLM conversation
            conclusion.txt - Step final answer
        decisions.txt      - All decisions with reasoning
        summary.txt        - Execution summary
        response_classification.txt - Response classification logic
        result.txt         - Final result with score
        error.txt          - Errors if any
"""

import os
import json
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List


# Clear visual delimiters
SECTION_DELIM = "=" * 80
SUBSECTION_DELIM = "-" * 80


@dataclass
class TaskLogger:
    """Logger for a single task execution."""
    
    task_index: int
    task_text: str
    run_id: Optional[str] = None
    log_dir: str = field(init=False)
    
    # Timing
    _start_time: float = field(default_factory=time.time, init=False)
    _last_milestone: float = field(default_factory=time.time, init=False)
    _milestones: List[str] = field(default_factory=list, init=False)
    
    # Internal state
    _current_step: int = field(default=0, init=False)
    _current_step_dir: Optional[str] = field(default=None, init=False)
    _replan_count: int = field(default=0, init=False)
    
    def __post_init__(self):
        # Create log directory
        base_dir = os.path.join(os.path.dirname(__file__), "logs")
        if self.run_id:
            self.log_dir = os.path.join(base_dir, f"run_{self.run_id}", f"task_{self.task_index}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir = os.path.join(base_dir, f"task_{self.task_index}_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Write task with header
        task_content = f"""{SECTION_DELIM}
TASK {self.task_index}
{SECTION_DELIM}

{self.task_text}

{SUBSECTION_DELIM}
Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{SUBSECTION_DELIM}
"""
        self._write("task.txt", task_content)
        self._milestone("Started")
    
    def _write(self, filename: str, content: str) -> None:
        """Write content to file in log directory (immediate flush)."""
        path = os.path.join(self.log_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk
    
    def _append(self, filename: str, content: str) -> None:
        """Append content to file in log directory (immediate flush)."""
        path = os.path.join(self.log_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk
    
    def _milestone(self, message: str) -> None:
        """Record milestone."""
        now = time.time()
        elapsed = now - self._last_milestone
        total = now - self._start_time
        self._last_milestone = now
        
        # Store for timeline log
        self._milestones.append(f"[{total:6.1f}s +{elapsed:5.1f}s] {message}")
        
        # Minimal bash output - just dots for progress
        print(".", end="", flush=True)
    
    def _format_timestamp(self) -> str:
        """Get formatted timestamp."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # ========================================
    # Wiki change logging
    # ========================================
    
    def log_wiki_change(self, old_sha: str, new_sha: str, old_rules: str = None, new_rules: str = None) -> None:
        """Log a wiki change event with before/after states."""
        timestamp = self._format_timestamp()
        
        content = f"""{SECTION_DELIM}
WIKI CHANGE DETECTED
{SECTION_DELIM}
Time: {timestamp}
Old SHA: {old_sha or 'none'}
New SHA: {new_sha}

"""
        if old_rules:
            content += f"""{SUBSECTION_DELIM}
OLD RULES (SHA: {old_sha[:16] if old_sha else 'none'}...)
{SUBSECTION_DELIM}
{old_rules}

"""
        
        if new_rules:
            content += f"""{SUBSECTION_DELIM}
NEW RULES (SHA: {new_sha[:16]}...)
{SUBSECTION_DELIM}
{new_rules}
"""
        
        # Append to wiki_changes.txt to keep history
        self._append("wiki_changes.txt", content + "\n")
        self._milestone(f"Wiki changed: {old_sha[:8] if old_sha else 'none'} → {new_sha[:8]}")
    
    def log_wiki_initial(self, sha: str, rules: str) -> None:
        """Log the initial wiki state at task start."""
        timestamp = self._format_timestamp()
        
        content = f"""{SECTION_DELIM}
WIKI STATE (INITIAL)
{SECTION_DELIM}
Time: {timestamp}
SHA: {sha}

{SUBSECTION_DELIM}
RULES
{SUBSECTION_DELIM}
{rules}
"""
        self._write("wiki_changes.txt", content + "\n")
    
    # ========================================
    # Preflight logging
    # ========================================
    
    def log_preflight(self, should_proceed: bool, explanation: str = None, 
                      denial_outcome: str = None, denial_message: str = None) -> None:
        """Log the preflight check result."""
        status = "PASSED" if should_proceed else "DENIED"
        
        content = f"""{SECTION_DELIM}
PREFLIGHT CHECK
{SECTION_DELIM}
Time: {self._format_timestamp()}
Status: {status}
"""
        if explanation:
            content += f"Explanation: {explanation}\n"
        
        if not should_proceed:
            content += f"""
{SUBSECTION_DELIM}
DENIAL DETAILS
{SUBSECTION_DELIM}
Outcome: {denial_outcome}
Message: {denial_message}
"""
        
        self._write("preflight.txt", content)
        self._milestone(f"Preflight {status}")
    
    # ========================================
    # Plan logging
    # ========================================
    
    def log_plan(self, plan_text: str) -> None:
        """Log the execution plan."""
        content = f"""{SECTION_DELIM}
INITIAL PLAN
{SECTION_DELIM}
Created: {self._format_timestamp()}

{plan_text}
"""
        self._write("plan.txt", content)
        self._milestone("Plan created")
    
    def log_replan(self, replan_num: int, replan_type: str, plan_text: str) -> None:
        """Log a replan (appends to plan.txt)."""
        self._replan_count = replan_num
        content = f"""

{SECTION_DELIM}
REPLAN #{replan_num} ({replan_type})
{SECTION_DELIM}
Created: {self._format_timestamp()}

{plan_text}
"""
        self._append("plan.txt", content)
        self._milestone(f"Replan #{replan_num}")
    
    # ========================================
    # Step logging
    # ========================================
    
    def start_step(self, step_num: int, step_description: str) -> None:
        """Start logging a new step."""
        self._current_step = step_num
        self._current_step_dir = f"step_{step_num}"
        
        # Create step directory
        step_path = os.path.join(self.log_dir, self._current_step_dir)
        os.makedirs(step_path, exist_ok=True)
        
        # Write prompt with header
        prompt_content = f"""{SECTION_DELIM}
STEP {step_num}
{SECTION_DELIM}
Started: {self._format_timestamp()}

{step_description}
"""
        self._write(f"{self._current_step_dir}/prompt.txt", prompt_content)
    
    def init_messages_file(self, system_prompt: str = None) -> None:
        """Initialize the messages file with header and system prompt."""
        if not self._current_step_dir:
            return
        
        lines = [
            SECTION_DELIM,
            f"STEP {self._current_step} - LLM CONVERSATION",
            SECTION_DELIM,
            f"Started: {self._format_timestamp()}",
            ""
        ]
        
        # Include system prompt if provided
        if system_prompt:
            lines.extend([
                SUBSECTION_DELIM,
                "[SYSTEM PROMPT]",
                SUBSECTION_DELIM,
                system_prompt,
                ""
            ])
        
        self._write(f"{self._current_step_dir}/messages.txt", "\n".join(lines))
        self._message_count = 0
    
    def append_message(self, role: str, content: str) -> None:
        """Append a single message to the messages file immediately."""
        if not self._current_step_dir:
            return
        
        self._message_count = getattr(self, '_message_count', 0) + 1
        
        lines = [
            SUBSECTION_DELIM,
            f"[{role.upper()}] (message {self._message_count})",
            SUBSECTION_DELIM,
            content,
            ""
        ]
        
        self._append(f"{self._current_step_dir}/messages.txt", "\n".join(lines))
    
    def log_reasoning(self, reasoning: str) -> None:
        """
        Log reasoning/thinking text to a separate file (NOT to messages).
        This keeps reasoning out of the conversation history but preserved in logs.
        """
        if not self._current_step_dir or not reasoning:
            return
        
        self._reasoning_count = getattr(self, '_reasoning_count', 0) + 1
        
        lines = [
            SUBSECTION_DELIM,
            f"[REASONING #{self._reasoning_count}] {self._format_timestamp()}",
            SUBSECTION_DELIM,
            reasoning,
            ""
        ]
        
        self._append(f"{self._current_step_dir}/reasoning.txt", "\n".join(lines))
    
    def log_messages(self, messages: list, system_prompt: str = None) -> None:
        """Log the full LLM conversation for current step (legacy, for compatibility)."""
        if not self._current_step_dir:
            return
        
        lines = [
            SECTION_DELIM,
            f"STEP {self._current_step} - LLM CONVERSATION",
            SECTION_DELIM,
            f"Logged: {self._format_timestamp()}",
            ""
        ]
        
        # Include system prompt if provided
        if system_prompt:
            lines.extend([
                SUBSECTION_DELIM,
                "[SYSTEM PROMPT]",
                SUBSECTION_DELIM,
                system_prompt,
                ""
            ])
        
        # Format each message with clear delimiters
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            
            lines.extend([
                SUBSECTION_DELIM,
                f"[{role}] (message {i+1})",
                SUBSECTION_DELIM,
                content,
                ""
            ])
        
        self._write(f"{self._current_step_dir}/messages.txt", "\n".join(lines))
    
    def log_step_conclusion(self, conclusion: str) -> None:
        """Log step conclusion/final answer."""
        if not self._current_step_dir:
            return
        
        content = f"""{SECTION_DELIM}
STEP {self._current_step} - CONCLUSION
{SECTION_DELIM}
Completed: {self._format_timestamp()}

{conclusion}
"""
        self._write(f"{self._current_step_dir}/conclusion.txt", content)
        self._milestone(f"Step {self._current_step}")
    
    # ========================================
    # Decision logging
    # ========================================
    
    def log_decision(self, step_num: int, decision: str, reasoning: str, context: str = None) -> None:
        """Log a decision after step completion."""
        entry = f"""{SUBSECTION_DELIM}
DECISION AFTER STEP {step_num}
{SUBSECTION_DELIM}
Time: {self._format_timestamp()}
Decision: {decision}

Reasoning:
{reasoning}
"""
        if context:
            entry += f"""
Context:
{context}
"""
        entry += "\n"
        self._append("decisions.txt", entry)
    
    # ========================================
    # Final logging
    # ========================================
    
    def log_result(self, score: float, outcome: str, message: str, 
                   eval_logs: str = None, links: list = None) -> None:
        """Log the final result."""
        content = f"""{SECTION_DELIM}
TASK {self.task_index} - FINAL RESULT
{SECTION_DELIM}
Completed: {self._format_timestamp()}
Total Time: {time.time() - self._start_time:.1f}s

{SUBSECTION_DELIM}
SCORE: {score}
{SUBSECTION_DELIM}

Outcome: {outcome}
Message: {message}
"""
        if links:
            content += f"Links:\n{json.dumps(links, indent=4)}\n"
        
        if eval_logs:
            content += f"""
{SUBSECTION_DELIM}
EVALUATION LOGS
{SUBSECTION_DELIM}
{eval_logs}
"""
        
        # Add timeline
        content += f"""
{SUBSECTION_DELIM}
EXECUTION TIMELINE
{SUBSECTION_DELIM}
"""
        for m in self._milestones:
            content += f"{m}\n"
        
        self._write("result.txt", content)
    
    def log_summary(self, summary: str) -> None:
        """Log the execution summary."""
        content = f"""{SECTION_DELIM}
EXECUTION SUMMARY
{SECTION_DELIM}
Generated: {self._format_timestamp()}

{summary}
"""
        self._write("summary.txt", content)
        self._milestone("Finished")
    
    def log_error(self, error: str, traceback: str = None) -> None:
        """Log an error."""
        content = f"""{SECTION_DELIM}
ERROR
{SECTION_DELIM}
Time: {self._format_timestamp()}

{error}
"""
        if traceback:
            content += f"""
{SUBSECTION_DELIM}
TRACEBACK
{SUBSECTION_DELIM}
{traceback}
"""
        self._write("error.txt", content)
        self._milestone("ERROR")
    
    # ========================================
    # Response Classification logging
    # ========================================
    
    def log_response_classification(
        self,
        agent_answer: str,
        classification_checks: dict,
        selected_outcome: str,
        final_message: str,
        links: list = None,
        full_prompt: str = None,
        task: str = None,
        user_context: str = None,
        wiki_rules: str = None,
        company_policies_check: str = None
    ) -> None:
        """
        Log the response classification process.
        
        Args:
            agent_answer: The agent's execution summary (input)
            classification_checks: Dict of {check_name: {applies: bool, reasoning: str, erc_code: str}}
            selected_outcome: The final ERC3 outcome code
            final_message: The response message
            links: Optional list of entity links
            full_prompt: The complete prompt sent to LLM
            task: Original task text
            user_context: User context string
            wiki_rules: Wiki rules used
            company_policies_check: Result of company policies field
        """
        lines = [
            SECTION_DELIM,
            "RESPONSE CLASSIFICATION",
            SECTION_DELIM,
            f"Time: {self._format_timestamp()}",
            "",
            SUBSECTION_DELIM,
            "INPUT: TASK",
            SUBSECTION_DELIM,
            task or "(not provided)",
            "",
            SUBSECTION_DELIM,
            "INPUT: USER CONTEXT",
            SUBSECTION_DELIM,
            user_context or "(not provided)",
            "",
            SUBSECTION_DELIM,
            "INPUT: WIKI RULES",
            SUBSECTION_DELIM,
            wiki_rules or "(not provided)",
            "",
            SUBSECTION_DELIM,
            "INPUT: AGENT ANSWER",
            SUBSECTION_DELIM,
            agent_answer,
            "",
            SUBSECTION_DELIM,
            "COMPANY POLICIES CHECK",
            SUBSECTION_DELIM,
            company_policies_check or "(not evaluated)",
            "",
            SUBSECTION_DELIM,
            "CLASSIFICATION CHECKS (in priority order)",
            SUBSECTION_DELIM,
        ]
        
        # Add each check with its reasoning
        for check_name, check_data in classification_checks.items():
            applies = check_data.get("applies", False)
            reasoning = check_data.get("reasoning", "")
            erc_code = check_data.get("erc_code", "")
            
            status = "✓ APPLIES" if applies else "✗ does not apply"
            lines.append(f"\n{check_name} → {erc_code}")
            lines.append(f"  Status: {status}")
            lines.append(f"  Reasoning: {reasoning}")
        
        # Add selection logic
        lines.extend([
            "",
            SUBSECTION_DELIM,
            "OUTCOME SELECTION",
            SUBSECTION_DELIM,
            "Priority order: functionality_not_available > permission_denied > more_info_needed > system_error > object_not_found > task_completed",
            "",
            f"Selected outcome: {selected_outcome}",
            "",
            SUBSECTION_DELIM,
            "FINAL RESPONSE",
            SUBSECTION_DELIM,
            f"Outcome: {selected_outcome}",
            f"Message: {final_message}",
        ])
        
        if links:
            lines.append(f"Links: {json.dumps(links, indent=2)}")
        
        # Also save the full prompt for debugging
        if full_prompt:
            lines.extend([
                "",
                SUBSECTION_DELIM,
                "FULL PROMPT (sent to LLM)",
                SUBSECTION_DELIM,
                full_prompt
            ])
        
        self._write("response_classification.txt", "\n".join(lines))
    
    def get_elapsed_time(self) -> float:
        """Get total elapsed time."""
        return time.time() - self._start_time
