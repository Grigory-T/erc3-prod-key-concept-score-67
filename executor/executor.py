"""
Python Code Executor - runs code in isolated environment with helper functions.

NO sub-agents. Just code execution.
"""

import ast
import contextlib
import io
import linecache
import multiprocessing
import traceback
from contextlib import suppress
from typing import Any, Dict, Optional

import threading

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Python Code Executor")
WORKERS: Dict[str, "Worker"] = {}
_workers_lock = threading.Lock()


class Payload(BaseModel):
    code: str
    interpreter_id: str = "default"
    session_id: Optional[str] = None
    task_id: Optional[str] = None


class ExecResult(BaseModel):
    stdout: str
    error: str
    result: Any | None = None


def _exec_with_return(code: str, globals_dict: Dict[str, Any], filename: str = "snippet.py") -> Any:
    linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
    mod = ast.parse(code, filename=filename, mode="exec")

    last_expr_node = None
    last_assigned_name: str | None = None

    if mod.body:
        last = mod.body[-1]
        if isinstance(last, ast.Expr):
            last_expr_node = last.value
            mod.body.pop()
        elif isinstance(last, ast.Assign) and last.targets and isinstance(last.targets[0], ast.Name):
            last_assigned_name = last.targets[0].id
        elif isinstance(last, ast.AnnAssign) and last.value is not None and isinstance(last.target, ast.Name):
            last_assigned_name = last.target.id
        elif isinstance(last, ast.AugAssign) and isinstance(last.target, ast.Name):
            last_assigned_name = last.target.id

    exec_code = compile(mod, filename, "exec")
    globals_dict.setdefault("__name__", "__main__")
    globals_dict.setdefault("__file__", filename)
    exec(exec_code, globals_dict, None)

    if last_expr_node is not None:
        expr_ast = ast.Expression(last_expr_node)
        ast.fix_missing_locations(expr_ast)
        eval_code = compile(expr_ast, filename, "eval")
        return eval(eval_code, globals_dict, None)

    if last_assigned_name is not None:
        return globals_dict.get(last_assigned_name)

    return None


def _execute_code(code: str, globals_dict: Dict[str, Any], filename: str = "snippet.py") -> ExecResult:
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            result = _exec_with_return(code, globals_dict, filename=filename)
        return ExecResult(stdout=buf.getvalue(), error="", result=result)
    except SyntaxError as e:
        line = e.text if e.text is not None else linecache.getline(e.filename or filename, e.lineno or 0)
        caret = " " * (e.offset - 1) + "^" if e.offset and e.offset > 0 else ""
        msg = f"{e.filename}:{e.lineno}:{e.offset}: {e.msg}\n{line}{caret}"
        return ExecResult(stdout=buf.getvalue(), error=msg, result=None)
    except Exception as e:
        tb = traceback.TracebackException.from_exception(e)
        # Filter frames from any snippet_*.py file (not just current one)
        filtered_frames = [f for f in tb.stack if f.filename.startswith("snippet")]
        if filtered_frames:
            tb.stack = type(tb.stack).from_list(filtered_frames)
        return ExecResult(stdout=buf.getvalue(), error="".join(tb.format()), result=None)


def _worker_main(request_q: multiprocessing.Queue, response_q: multiprocessing.Queue) -> None:
    import math
    import re
    import itertools
    import more_itertools as mit
    from pydantic import BaseModel
    from erc3 import ERC3, erc3 as dev
    import difflib
    
    # Import wrapper functions for erc3-dev benchmark (NO delegation)
    from dev_functions import (
        _init_dev_api,
        # Types
        SkillLevel, EmployeeBrief, EmployeeFull, CustomerBrief, CustomerFull,
        ProjectBrief, ProjectFull, TeamMember, TimeEntry, WhoAmI,
        TimeSummaryByProject, TimeSummaryByEmployee,
        # Who am I
        who_am_i,
        # Employees
        list_employees, list_all_employees, get_employee, search_employees,
        # Customers
        list_customers, list_all_customers, get_customer, search_customers,
        # Projects
        list_projects, list_all_projects, get_project, search_projects,
        # Time entries
        search_time_entries, get_time_entry, log_time_entry, update_time_entry,
        time_summary_by_project, time_summary_by_employee,
        # Wiki
        list_wiki, load_wiki, search_wiki, search_wiki_fuzzy, update_wiki,
        # Fuzzy matching
        fuzzy_compare, fuzzy_find_in_text,
        # Updates
        update_employee_info, update_project_status, update_project_team,
    )

    core = ERC3()

    _globals: Dict[str, Any] = {
        "__name__": "__main__",
        "math": math,
        "re": re,
        "itertools": itertools,
        "it": itertools,
        "more_itertools": mit,
        "mit": mit,
        "BaseModel": BaseModel,
        "core": core,
        "dev": dev,
        "difflib": difflib,
        # Type definitions
        "SkillLevel": SkillLevel,
        "EmployeeBrief": EmployeeBrief,
        "EmployeeFull": EmployeeFull,
        "CustomerBrief": CustomerBrief,
        "CustomerFull": CustomerFull,
        "ProjectBrief": ProjectBrief,
        "ProjectFull": ProjectFull,
        "TeamMember": TeamMember,
        "TimeEntry": TimeEntry,
        "WhoAmI": WhoAmI,
        "TimeSummaryByProject": TimeSummaryByProject,
        "TimeSummaryByEmployee": TimeSummaryByEmployee,
        # Who am I
        "who_am_i": who_am_i,
        # Employee functions
        "list_employees": list_employees,
        "list_all_employees": list_all_employees,
        "get_employee": get_employee,
        "search_employees": search_employees,
        # Customer functions
        "list_customers": list_customers,
        "list_all_customers": list_all_customers,
        "get_customer": get_customer,
        "search_customers": search_customers,
        # Project functions
        "list_projects": list_projects,
        "list_all_projects": list_all_projects,
        "get_project": get_project,
        "search_projects": search_projects,
        # Time entry functions
        "search_time_entries": search_time_entries,
        "get_time_entry": get_time_entry,
        "log_time_entry": log_time_entry,
        "update_time_entry": update_time_entry,
        "time_summary_by_project": time_summary_by_project,
        "time_summary_by_employee": time_summary_by_employee,
        # Wiki functions
        "list_wiki": list_wiki,
        "load_wiki": load_wiki,
        "search_wiki": search_wiki,
        "search_wiki_fuzzy": search_wiki_fuzzy,
        "update_wiki": update_wiki,
        # Fuzzy matching
        "fuzzy_compare": fuzzy_compare,
        "fuzzy_find_in_text": fuzzy_find_in_text,
        # Update functions
        "update_employee_info": update_employee_info,
        "update_project_status": update_project_status,
        "update_project_team": update_project_team,
        # NOTE: provide_response is NOT exposed - handled separately in runner
        # NOTE: delegate_to_agent REMOVED - linear agent only
    }

    _exec_counter = 0

    while True:
        item = request_q.get()
        if item is None:
            break

        code: str = item["code"]
        session_id: Optional[str] = item.get("session_id")
        task_id: Optional[str] = item.get("task_id")

        _exec_counter += 1
        filename = f"snippet_{_exec_counter}.py"

        if session_id and task_id:
            try:
                status = core.session_status(session_id)
                task = next((t for t in status.tasks if t.task_id == task_id), None)
                if task:
                    dev_api = core.get_erc_dev_client(task)
                    _globals["task"] = task
                    _globals["dev_api"] = dev_api
                    _globals["status"] = status
                    # Initialize wrapper functions with dev_api
                    _init_dev_api(dev_api, dev)
            except Exception as e:
                response_q.put(ExecResult(stdout="", error=str(e), result=None).model_dump())
                continue

        exec_result = _execute_code(code, _globals, filename=filename)
        response_q.put(exec_result.model_dump())


class Worker:
    def __init__(self) -> None:
        self._request_q: multiprocessing.Queue = multiprocessing.Queue()
        self._response_q: multiprocessing.Queue = multiprocessing.Queue()
        self._exec_lock = threading.Lock()  # Ensures request-response pairing
        self._process = multiprocessing.Process(target=_worker_main, args=(self._request_q, self._response_q), daemon=True)
        self._process.start()

    def exec(self, code: str, session_id: Optional[str], task_id: Optional[str]) -> dict:
        """Thread-safe execution - ensures request/response are paired."""
        with self._exec_lock:
            # Check if subprocess is alive, restart if needed
            if not self._process.is_alive():
                print(f"[EXECUTOR] Worker subprocess died, restarting...")
                # Clean up old queues
                with suppress(Exception):
                    self._request_q.close()
                with suppress(Exception):
                    self._response_q.close()
                # Create new queues and process
                self._request_q = multiprocessing.Queue()
                self._response_q = multiprocessing.Queue()
                self._process = multiprocessing.Process(
                    target=_worker_main, 
                    args=(self._request_q, self._response_q), 
                    daemon=True
                )
                self._process.start()
            
            self._request_q.put({
                "code": code, 
                "session_id": session_id, 
                "task_id": task_id
            })
            
            # Timeout to prevent indefinite hang if something goes wrong
            try:
                return self._response_q.get(timeout=300)  # 5 minute timeout
            except Exception:
                # If timeout or error, check if process died
                if not self._process.is_alive():
                    return {"stdout": "", "error": "Worker subprocess died during execution", "result": None}
                raise

    def terminate(self) -> None:
        self._request_q.put(None)
        if self._process.is_alive():
            with suppress(Exception):
                self._process.terminate()
            self._process.join(timeout=2)
        if self._process.is_alive():
            with suppress(Exception):
                self._process.kill()
            self._process.join(timeout=1)
        with suppress(Exception):
            self._request_q.close()
        with suppress(Exception):
            self._response_q.close()


def _get_worker(interpreter_id: str) -> Worker:
    """Thread-safe worker retrieval/creation."""
    with _workers_lock:
        if interpreter_id not in WORKERS:
            WORKERS[interpreter_id] = Worker()
        return WORKERS[interpreter_id]


@app.post("/exec")
def exec_code(payload: Payload) -> ExecResult:
    worker = _get_worker(payload.interpreter_id)
    result_dict = worker.exec(
        payload.code, 
        payload.session_id, 
        payload.task_id
    )
    return ExecResult.model_validate(result_dict)


@app.post("/cleanup")
def cleanup(interpreter_id: str = "default") -> dict:
    """Thread-safe worker cleanup."""
    with _workers_lock:
        worker = WORKERS.pop(interpreter_id, None)
    if worker:
        worker.terminate()
    return {"status": "ok"}
