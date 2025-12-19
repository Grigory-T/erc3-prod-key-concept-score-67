"""
ERC3 Runner - Multiprocessing Task Executor

Runs agent tasks in parallel using multiple processes.
Supports ~100+ tasks with configurable worker count.
"""

import os
import sys
import time
import traceback
import multiprocessing as mp
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from queue import Empty

from erc3 import ERC3, erc3 as dev

from .agent import run_agent, AgentResult
from .response import decide_response, ResponseDecision
from .utils import set_llm_logger
from .wiki_rules import create_wiki_rules_context


# ============================================================
# CONFIGURATION
# ============================================================

# Number of parallel worker processes (3-5 recommended for prod)
NUM_WORKERS = 5

# Set to specific indices for testing, or None to run ALL tasks (for prod)
TASK_INDICES = []  # [0, 1, 2]  # Testing subset
# TASK_INDICES = range(0, 16)  # Testing range
# TASK_INDICES = None  # None = run all tasks from session (for prod)

# Session configuration
# Set to None to create new session, or paste existing session_id to resume
SESSION_ID = None

# Benchmark to use
BENCHMARK = "erc3-prod"  # -dev -test -prod

# Competition flags
FLAGS = ["compete_budget"]  # ["compete_accuracy"] for accuracy competition

# Maximum retries for failed tasks
MAX_TASK_RETRIES = 2

# ============================================================


def submit_response(dev_api, decision: ResponseDecision) -> None:
    """Submit the response via ERC3 API."""
    kwargs = {
        "message": decision.message,
        "outcome": decision.outcome
    }
    if decision.links:
        kwargs["links"] = [dev.AgentLink(kind=l.kind, id=l.id) for l in decision.links]
    
    dev_api.dispatch(dev.Req_ProvideAgentResponse(**kwargs))


def run_single_task(session_id: str, task_index: int, run_id: str) -> dict:
    """
    Run a single task. Returns dict with results.
    This function runs in a worker process.
    """
    task_start = time.time()
    pid = os.getpid()
    
    try:
        # Create ERC3 client (each process needs its own)
        core = ERC3()
        
        # Get task info
        status = core.session_status(session_id)
        task_info = status.tasks[task_index]
        task_id = task_info.task_id
        task_text = task_info.task_text
        
        print(f"[DEBUG] Task {task_index}: status={task_info.status}, task_id={task_id}", flush=True)
        
        # Complete in-progress tasks (important for resuming)
        if task_info.status == "in_progress":
            print(f"[DEBUG] Task {task_index}: completing in_progress task...", flush=True)
            result = core.complete_task(task_info)
            print(f"[DEBUG] Task {task_index}: complete_task result: {result}", flush=True)
        
        # Start task
        print(f"[DEBUG] Task {task_index}: calling start_task...", flush=True)
        start_result = core.start_task(task_info)
        print(f"[DEBUG] Task {task_index}: start_task result: {start_result}", flush=True)
        
        # Small delay to let simulation initialize
        time.sleep(0.5)
        
        # Refresh task_info to get new simulation credentials
        print(f"[DEBUG] Task {task_index}: refreshing task_info...", flush=True)
        status = core.session_status(session_id)
        task_info = status.tasks[task_index]
        print(f"[DEBUG] Task {task_index}: new status={task_info.status}", flush=True)
        
        # Print task_info details for debugging
        print(f"[DEBUG] Task {task_index}: task_info attrs: {[a for a in dir(task_info) if not a.startswith('_')]}", flush=True)
        
        print(f"[DEBUG] Task {task_index}: getting dev_api client...", flush=True)
        dev_api = core.get_erc_dev_client(task_info)
        print(f"[DEBUG] Task {task_index}: dev_api obtained", flush=True)
        
        # Set up LLM logging for this task
        set_llm_logger(core.log_llm, task_id)
        
        # Get user context for response decision
        # Use placeholder values - agent will call who_am_i() through executor
        user_context_dict = {
            "current_user": "unknown",
            "department": "unknown",
            "location": "unknown",
            "today": "unknown",
            "is_public": False,
            "wiki_sha1": "unknown"
        }
        print(f"[DEBUG] Task {task_index}: using placeholder user_context (agent will call who_am_i)", flush=True)
        
        # Create wiki rules context (with file-based locking for cache writes)
        wiki_rules_ctx = create_wiki_rules_context(dev_api, user_context_dict)
        wiki_rules = wiki_rules_ctx.get_formatted_rules()
        
        # Run agent
        agent_result = run_agent(
            task=task_text,
            session_id=session_id,
            task_id=task_id,
            task_index=task_index,
            run_id=run_id,
            user_context=user_context_dict,
            dev_api=dev_api,
            wiki_rules=wiki_rules
        )
        
        # Use agent's final wiki_rules and user_context
        final_wiki_rules = agent_result.wiki_rules
        final_user_context = agent_result.user_context
        
        # Refresh wiki one more time before response classification
        try:
            current_who = dev_api.who_am_i()
            if current_who.wiki_sha1 and current_who.wiki_sha1 != final_user_context.get("wiki_sha1"):
                final_user_context["wiki_sha1"] = current_who.wiki_sha1
                final_user_context["today"] = current_who.today or final_user_context.get("today")
                wiki_rules_ctx = create_wiki_rules_context(dev_api, final_user_context)
                final_wiki_rules = wiki_rules_ctx.get_formatted_rules()
        except Exception:
            pass  # Skip wiki refresh if who_am_i fails
        
        # Decide response
        decision = decide_response(
            task_text, 
            agent_result.answer, 
            final_user_context, 
            wiki_rules=final_wiki_rules, 
            logger=agent_result.log
        )
        
        # Submit response via erc3
        submit_response(dev_api, decision)
        
        # Get evaluation
        eval_result = core.complete_task(task_info)
        
        # Get score and log result to file
        score = None
        eval_logs = None
        if eval_result.eval:
            score = eval_result.eval.score
            eval_logs = eval_result.eval.logs
        
        # Log the final result
        agent_result.log.log_result(
            score=score,
            outcome=decision.outcome,
            message=decision.message,
            eval_logs=eval_logs,
            links=[{"kind": l.kind, "id": l.id} for l in decision.links] if decision.links else None
        )
        
        elapsed = time.time() - task_start
        
        return {
            "task_index": task_index,
            "score": score,
            "outcome": decision.outcome,
            "time_seconds": elapsed,
            "error": None,
            "pid": pid
        }
        
    except Exception as e:
        elapsed = time.time() - task_start
        error_msg = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc()
        
        return {
            "task_index": task_index,
            "score": None,
            "outcome": "error_internal",
            "time_seconds": elapsed,
            "error": error_msg,
            "traceback": tb,
            "pid": pid
        }


def worker_process(task_queue: mp.Queue, result_queue: mp.Queue, 
                   session_id: str, run_id: str, worker_id: int):
    """
    Worker process that pulls tasks from queue and executes them.
    """
    pid = os.getpid()
    print(f"[Worker {worker_id}] Started (PID: {pid})", flush=True)
    
    while True:
        try:
            # Get next task (with timeout to allow clean shutdown)
            try:
                task_index = task_queue.get(timeout=2)
            except Empty:
                continue
            
            # Poison pill - time to exit
            if task_index is None:
                print(f"[Worker {worker_id}] Shutting down", flush=True)
                break
            
            print(f"[Worker {worker_id}] Starting task {task_index}", flush=True)
            
            # Run the task
            result = run_single_task(
                session_id=session_id,
                task_index=task_index,
                run_id=run_id
            )
            
            # Put result in result queue
            result_queue.put(result)
            
            status = "✓" if result.get("score") and result["score"] >= 0.9 else "✗"
            print(f"[Worker {worker_id}] Finished task {task_index}: {status} ({result['time_seconds']:.0f}s)", flush=True)
            
        except Exception as e:
            # Unexpected error in worker loop
            print(f"[Worker {worker_id}] ERROR: {e}", flush=True)
            traceback.print_exc()
            continue


def run_parallel(session_id: str, task_indices: List[int], run_id: str,
                 num_workers: int = NUM_WORKERS) -> List[dict]:
    """
    Run tasks in parallel using worker pool.
    """
    print(f"\nStarting {num_workers} workers for {len(task_indices)} tasks...")
    
    # Create queues
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # Start worker processes FIRST
    workers = []
    for i in range(num_workers):
        p = mp.Process(
            target=worker_process,
            args=(task_queue, result_queue, session_id, run_id, i),
            daemon=False  # Don't use daemon - we want clean shutdown
        )
        p.start()
        workers.append(p)
    
    # Give workers time to start
    time.sleep(0.5)
    
    # Add all tasks to queue (in order)
    for task_idx in task_indices:
        task_queue.put(task_idx)
    
    # Add poison pills for workers to exit (after all tasks)
    for _ in range(num_workers):
        task_queue.put(None)
    
    start_time = time.time()
    
    # Collect results
    results = {}
    retry_counts = {idx: 0 for idx in task_indices}
    tasks_to_retry = []
    
    total_tasks = len(task_indices)
    received = 0
    
    print(f"Waiting for {total_tasks} task results...", flush=True)
    
    while received < total_tasks:
        try:
            result = result_queue.get(timeout=300)  # 5 min timeout per result
            received += 1
            
            task_idx = result["task_index"]
            
            # Check if we need to retry
            if result.get("error") and retry_counts[task_idx] < MAX_TASK_RETRIES:
                retry_counts[task_idx] += 1
                tasks_to_retry.append(task_idx)
                print(f"[Main] Task {task_idx} failed, will retry ({retry_counts[task_idx]}/{MAX_TASK_RETRIES})")
                print(f"       Error: {result.get('error')}")
            else:
                results[task_idx] = result
                
                if result.get("error"):
                    print(f"[Main] Task {task_idx} FAILED after {retry_counts[task_idx]} retries")
                else:
                    score = result.get("score", 0) or 0
                    elapsed = time.time() - start_time
                    done = len(results)
                    total_score = sum((r.get("score") or 0) for r in results.values())
                    print(f"[Main] Progress: {done}/{total_tasks} | Score: {total_score:.1f} | Time: {elapsed:.0f}s")
            
            # If we have retries, add them back to the queue
            if tasks_to_retry and received == total_tasks:
                for retry_idx in tasks_to_retry:
                    task_queue.put(retry_idx)
                    total_tasks += 1
                tasks_to_retry.clear()
                
        except Empty:
            # Check if all workers are still alive
            alive = sum(1 for w in workers if w.is_alive())
            if alive == 0:
                print("[Main] All workers died unexpectedly!")
                break
            print(f"[Main] Waiting... ({alive} workers alive)", flush=True)
    
    # Wait for workers to finish
    print("\nWaiting for workers to finish...")
    for i, w in enumerate(workers):
        w.join(timeout=10)
        if w.is_alive():
            print(f"[Main] Force terminating worker {i}")
            w.terminate()
            w.join(timeout=5)
    
    # Return results sorted by task index
    return [results.get(idx, {"task_index": idx, "error": "No result"}) 
            for idx in task_indices if idx in results]


def run_sequential(session_id: str, task_indices: List[int], run_id: str) -> List[dict]:
    """
    Run tasks sequentially (fallback mode).
    """
    results = []
    for i, task_idx in enumerate(task_indices):
        print(f"[{i+1}/{len(task_indices)}] Task {task_idx}...", end="", flush=True)
        result = run_single_task(session_id, task_idx, run_id)
        results.append(result)
        
        status = "✓" if result.get("score") and result["score"] >= 0.9 else "✗"
        print(f" {status} ({result['time_seconds']:.0f}s)")
        
        if result.get("error"):
            print(f"    Error: {result['error']}")
    
    return results


# ============================================================
# Main execution
# ============================================================

if __name__ == "__main__":
    # Use spawn method for multiprocessing (more compatible)
    mp.set_start_method('spawn', force=True)
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_start = time.time()
    
    # Initialize ERC3
    core = ERC3()
    
    # Create or use existing session
    if SESSION_ID:
        session_id = SESSION_ID
        print(f"Using existing session: {session_id}")
    else:
        res = core.start_session(
            benchmark=BENCHMARK,
            workspace="my",
            name="key_concept_parallel",
            architecture="plan_execute_agent_mp",
            flags=FLAGS
        )
        session_id = res.session_id
        print(f"Created new session: {session_id}")
    
    # Determine which tasks to run
    status = core.session_status(session_id)
    total_tasks = len(status.tasks)
    
    if TASK_INDICES is not None:
        task_indices = list(TASK_INDICES)
    else:
        task_indices = list(range(total_tasks))
    
    print(f"\n{'='*60}")
    print(f"ERC3 Runner")
    print(f"{'='*60}")
    print(f"Run ID:     {run_id}")
    print(f"Session:    {session_id}")
    print(f"Benchmark:  {BENCHMARK}")
    print(f"Workers:    {NUM_WORKERS}")
    print(f"Tasks:      {len(task_indices)} of {total_tasks}")
    print(f"{'='*60}")
    
    # Run tasks
    if NUM_WORKERS > 1:
        results = run_parallel(session_id, task_indices, run_id, NUM_WORKERS)
    else:
        results = run_sequential(session_id, task_indices, run_id)
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    total_score = sum((r.get("score") or 0) for r in results)
    total_time = time.time() - run_start
    successes = sum(1 for r in results if not r.get("error"))
    failures = sum(1 for r in results if r.get("error"))
    
    print(f"Total Score:  {total_score:.1f}/{len(results)}")
    print(f"Success Rate: {successes}/{len(results)} ({100*successes/max(len(results),1):.1f}%)")
    print(f"Failures:     {failures}")
    print(f"Total Time:   {total_time/60:.1f} minutes")
    if results:
        print(f"Avg per Task: {total_time/len(results):.1f}s")
    
    # Show failed tasks
    failed_tasks = [r for r in results if r.get("error")]
    if failed_tasks:
        print(f"\nFailed tasks:")
        for r in failed_tasks:
            print(f"  Task {r['task_index']}: {r.get('error', 'Unknown')[:80]}")
    
    # Show scores by task
    print(f"\nScores by task:")
    for r in sorted(results, key=lambda x: x["task_index"]):
        score = r.get("score")
        status = "✓" if score and score >= 0.9 else ("✗" if r.get("error") else "○")
        score_str = f"{score:.2f}" if score is not None else "ERR"
        time_str = f"{r.get('time_seconds', 0):.0f}s"
        print(f"  [{r['task_index']:3d}] {status} {score_str} ({time_str})")
    
    print(f"{'='*60}")
    
    # Submit session for competition (uncomment for prod)
    core.submit_session(session_id)
