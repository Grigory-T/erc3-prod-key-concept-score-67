#!/usr/bin/env python3
"""
Log Statistics Analyzer for Tasks 0-102
Generates a CSV file with comprehensive statistics for each task.
"""

import os
import re
import csv
from pathlib import Path


def count_plan_steps(plan_file_path):
    """Count the number of initial plan steps from plan.txt"""
    if not os.path.exists(plan_file_path):
        return 0
    
    try:
        with open(plan_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the Steps section and count numbered items
        steps_match = re.search(r'Steps:\s*\n(.*?)(?=\n\n|\Z)', content, re.DOTALL)
        if not steps_match:
            return 0
        
        steps_section = steps_match.group(1)
        # Count lines that start with a number followed by a period (e.g., "  1.", "  2.")
        step_lines = re.findall(r'^\s+\d+\.', steps_section, re.MULTILINE)
        return len(step_lines)
    except Exception as e:
        print(f"Error reading {plan_file_path}: {e}")
        return 0


def count_completed_steps(task_dir):
    """Count the number of step_N folders in the task directory"""
    if not os.path.exists(task_dir):
        return 0
    
    try:
        step_dirs = [d for d in os.listdir(task_dir) 
                     if os.path.isdir(os.path.join(task_dir, d)) and d.startswith('step_')]
        return len(step_dirs)
    except Exception as e:
        print(f"Error counting steps in {task_dir}: {e}")
        return 0


def count_replans(decisions_file_path):
    """Count the number of replans performed from decisions.txt"""
    if not os.path.exists(decisions_file_path):
        return 0
    
    try:
        with open(decisions_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count lines that contain "Decision: replan"
        replan_matches = re.findall(r'Decision:\s*replan', content, re.IGNORECASE)
        return len(replan_matches)
    except Exception as e:
        print(f"Error reading {decisions_file_path}: {e}")
        return 0


def get_total_time_minutes(result_file_path):
    """Extract total time in minutes from result.txt"""
    if not os.path.exists(result_file_path):
        return 0
    
    try:
        with open(result_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for "Total Time: XXXXs" or "Total Time: XXXXXs"
        time_match = re.search(r'Total Time:\s*([0-9.]+)s', content)
        if time_match:
            seconds = float(time_match.group(1))
            minutes = int(round(seconds / 60))
            return minutes
        return 0
    except Exception as e:
        print(f"Error reading {result_file_path}: {e}")
        return 0


def count_total_characters_in_messages(task_dir):
    """Count total characters in all messages.txt files in step_N folders"""
    if not os.path.exists(task_dir):
        return 0
    
    total_chars = 0
    try:
        # Find all step_N directories
        for item in os.listdir(task_dir):
            step_path = os.path.join(task_dir, item)
            if os.path.isdir(step_path) and item.startswith('step_'):
                messages_file = os.path.join(step_path, 'messages.txt')
                if os.path.exists(messages_file):
                    with open(messages_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        total_chars += len(content)
    except Exception as e:
        print(f"Error counting characters in {task_dir}: {e}")
    
    return total_chars


def get_preflight_status(preflight_file_path):
    """Extract preflight check status from preflight.txt"""
    if not os.path.exists(preflight_file_path):
        return "N/A"
    
    try:
        with open(preflight_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for "Status: PASSED" or "Status: FAILED"
        status_match = re.search(r'Status:\s*(\w+)', content)
        if status_match:
            return status_match.group(1)
        return "N/A"
    except Exception as e:
        print(f"Error reading {preflight_file_path}: {e}")
        return "ERROR"


def get_task_outcome(result_file_path):
    """Extract task outcome from result.txt"""
    if not os.path.exists(result_file_path):
        return "N/A"
    
    try:
        with open(result_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for "Outcome: xxx"
        outcome_match = re.search(r'Outcome:\s*(\S+)', content)
        if outcome_match:
            return outcome_match.group(1)
        return "N/A"
    except Exception as e:
        print(f"Error reading {result_file_path}: {e}")
        return "ERROR"


def analyze_task(task_number, logs_dir):
    """Analyze a single task and return statistics dictionary"""
    task_id = f"t{task_number:03d}"
    task_dir = os.path.join(logs_dir, f"task_{task_number}")
    
    # Gather all statistics
    stats = {
        'task_id': task_id,
        'initial_plan_steps': count_plan_steps(os.path.join(task_dir, 'plan.txt')),
        'completed_steps': count_completed_steps(task_dir),
        'replans_count': count_replans(os.path.join(task_dir, 'decisions.txt')),
        'total_time_minutes': get_total_time_minutes(os.path.join(task_dir, 'result.txt')),
        'total_characters': count_total_characters_in_messages(task_dir),
        'preflight_status': get_preflight_status(os.path.join(task_dir, 'preflight.txt')),
        'task_outcome': get_task_outcome(os.path.join(task_dir, 'result.txt'))
    }
    
    return stats


def main():
    """Main function to analyze all tasks and generate CSV"""
    # Get the script directory
    script_dir = Path(__file__).parent
    logs_dir = script_dir / 'logs'
    output_csv = script_dir / 'logs_stats.csv'
    
    print("Starting log analysis...")
    print(f"Logs directory: {logs_dir}")
    print(f"Output CSV: {output_csv}")
    print()
    
    # Collect statistics for all tasks
    all_stats = []
    
    for task_num in range(103):  # Tasks 0-102
        print(f"Analyzing task {task_num}...", end=' ')
        stats = analyze_task(task_num, logs_dir)
        all_stats.append(stats)
        print(f"✓ ({stats['completed_steps']} steps, {stats['total_time_minutes']} min, {stats['task_outcome']})")
    
    # Write to CSV
    print()
    print("Writing CSV file...")
    
    fieldnames = [
        'task_id',
        'initial_plan_steps',
        'completed_steps',
        'replans_count',
        'total_time_minutes',
        'total_characters',
        'preflight_status',
        'task_outcome'
    ]
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_stats)
    
    print(f"✓ CSV file created: {output_csv}")
    print()
    
    # Print summary statistics
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    total_tasks = len(all_stats)
    total_steps = sum(s['completed_steps'] for s in all_stats)
    total_time = sum(s['total_time_minutes'] for s in all_stats)
    total_chars = sum(s['total_characters'] for s in all_stats)
    total_replans = sum(s['replans_count'] for s in all_stats)
    
    passed_preflight = sum(1 for s in all_stats if s['preflight_status'] == 'PASSED')
    ok_answers = sum(1 for s in all_stats if s['task_outcome'].startswith('ok'))
    
    print(f"Total tasks analyzed:     {total_tasks}")
    print(f"Total completed steps:    {total_steps}")
    print(f"Total replans:            {total_replans}")
    print(f"Total time (minutes):     {total_time:,}")
    print(f"Total time (hours):       {total_time/60:.1f}")
    print(f"Total characters:         {total_chars:,}")
    print(f"Preflight PASSED:         {passed_preflight}/{total_tasks}")
    print(f"Successful outcomes:      {ok_answers}/{total_tasks}")
    print()
    print(f"Average steps per task:   {total_steps/total_tasks:.1f}")
    print(f"Average time per task:    {total_time/total_tasks:.1f} min")
    print(f"Average chars per task:   {total_chars/total_tasks:,.0f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
