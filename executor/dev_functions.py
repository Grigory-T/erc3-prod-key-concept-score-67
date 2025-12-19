"""
Simple wrapper functions for ERC3-DEV API.

Usage:
    from dev_functions import init_dev, list_employees, get_employee, ...
    
    # Initialize once per task
    init_dev(task)
    
    # Then use any function
    employees = list_all_employees()
    emp = get_employee("elena_vogel")
"""

from typing import Optional, List
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from erc3 import ERC3, erc3 as dev

# Global state
_core: Optional[ERC3] = None
_dev_api = None


# ============================================================
# Type definitions (Pydantic models)
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
    deal_phase: str  # idea, exploring, active, paused, archived
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
    status: str  # idea, exploring, active, paused, archived


class TeamMember(BaseModel):
    """Project team member."""
    employee: str
    time_slice: float
    role: str  # Lead, Engineer, Designer, QA, Ops, Other


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
    status: str  # draft, submitted, approved, invoiced, voided


class WhoAmI(BaseModel):
    """Current user info."""
    current_user: Optional[str] = None
    is_public: bool = False
    location: Optional[str] = None
    department: Optional[str] = None
    today: Optional[str] = None
    wiki_sha1: Optional[str] = None


# ============================================================
# Initialization
# ============================================================

def init_dev(task) -> None:
    """
    Initialize dev API for a specific task.
    Must be called before using any other function.
    
    Args:
        task: TaskInfo object from ERC3
        
    Example:
        core = ERC3()
        status = core.session_status(session_id)
        task = status.tasks[0]
        init_dev(task)
    """
    global _core, _dev_api
    if _core is None:
        _core = ERC3()
    _dev_api = _core.get_erc_dev_client(task)


def get_core() -> ERC3:
    """Get ERC3 core instance."""
    global _core
    if _core is None:
        _core = ERC3()
    return _core


def get_api():
    """Get the raw dev API client."""
    return _dev_api


# ============================================================
# WHO AM I
# ============================================================

def who_am_i() -> WhoAmI:
    """
    Get current user context.
    
    Returns:
        WhoAmI with current_user, is_public, location, department, today, wiki_sha1
    """
    try:
        resp = _dev_api.who_am_i()
        return WhoAmI(
            current_user=resp.current_user,
            is_public=resp.is_public,
            location=resp.location,
            department=resp.department,
            today=resp.today,
            wiki_sha1=resp.wiki_sha1
        )
    except Exception as e:
        return WhoAmI(
            current_user="error_not_found",
            is_public=True,
            location="error_not_found",
            department="error_not_found",
            today="error_not_found",
            wiki_sha1="error_not_found"
        )


# ============================================================
# EMPLOYEES
# ============================================================

def list_employees(offset: int = 0, limit: int = 5) -> tuple[list[EmployeeBrief], int]:
    """
    Get a page of employees.
    
    Args:
        offset: Start position (0-based). Default 0.
        limit: Max items per page. MAX = 5! Default 5.
        
    Returns:
        Tuple of (employees_list, next_offset)
        - next_offset: Use this for next page, or -1 if no more data
    """
    resp = _dev_api.dispatch(dev.Req_ListEmployees(offset=offset, limit=limit))
    
    employees = []
    if resp.employees:
        for e in resp.employees:
            employees.append(EmployeeBrief(
                id=e.id,
                name=e.name,
                email=e.email,
                salary=e.salary,
                location=e.location,
                department=e.department
            ))
    
    next_offset = resp.next_offset if resp.next_offset is not None else -1
    return employees, next_offset


def list_all_employees() -> list[EmployeeBrief]:
    """Get ALL employees (handles pagination automatically)."""
    all_employees = []
    offset = 0
    
    while True:
        employees, next_offset = list_employees(offset=offset, limit=5)
        all_employees.extend(employees)
        
        if next_offset < 0:
            break
        offset = next_offset
    
    return all_employees


def get_employee(employee_id: str) -> Optional[EmployeeFull]:
    """
    Get full employee details.
    
    Args:
        employee_id: Employee ID string
        
    Returns:
        EmployeeFull or None if not found
    """
    try:
        resp = _dev_api.dispatch(dev.Req_GetEmployee(id=employee_id))
        if resp.employee:
            e = resp.employee
            return EmployeeFull(
                id=e.id,
                name=e.name,
                email=e.email,
                salary=e.salary,
                notes=e.notes,
                location=e.location,
                department=e.department,
                skills=[SkillLevel(name=s.name, level=s.level) for s in (e.skills or [])],
                wills=[SkillLevel(name=w.name, level=w.level) for w in (e.wills or [])]
            )
    except:
        pass
    return None


def search_employees(query: str = None, location: str = None, department: str = None,
                     skills: list = None, offset: int = 0, limit: int = 5) -> tuple[list[EmployeeBrief], int]:
    """
    Search employees with filters.
    
    Args:
        query: Search query string
        location: Filter by location
        department: Filter by department
        skills: List of skill filters [{name, min_level}]
        offset: Pagination offset
        limit: Page size (max 5)
        
    Returns:
        Tuple of (employees_list, next_offset)
    """
    kwargs = {"offset": offset, "limit": limit}
    if query:
        kwargs["query"] = query
    if location:
        kwargs["location"] = location
    if department:
        kwargs["department"] = department
    if skills:
        kwargs["skills"] = [dev.SkillFilter(**s) for s in skills]
    
    resp = _dev_api.dispatch(dev.Req_SearchEmployees(**kwargs))
    
    employees = []
    if resp.employees:
        for e in resp.employees:
            employees.append(EmployeeBrief(
                id=e.id,
                name=e.name,
                email=e.email,
                salary=e.salary,
                location=e.location,
                department=e.department
            ))
    
    next_offset = resp.next_offset if resp.next_offset is not None else -1
    return employees, next_offset


# ============================================================
# CUSTOMERS
# ============================================================

def list_customers(offset: int = 0, limit: int = 5) -> tuple[list[CustomerBrief], int]:
    """Get a page of customers."""
    resp = _dev_api.dispatch(dev.Req_ListCustomers(offset=offset, limit=limit))
    
    customers = []
    if resp.companies:
        for c in resp.companies:
            customers.append(CustomerBrief(
                id=c.id,
                name=c.name,
                location=c.location,
                deal_phase=c.deal_phase,
                high_level_status=c.high_level_status
            ))
    
    next_offset = resp.next_offset if resp.next_offset is not None else -1
    return customers, next_offset


def list_all_customers() -> list[CustomerBrief]:
    """Get ALL customers."""
    all_customers = []
    offset = 0
    
    while True:
        customers, next_offset = list_customers(offset=offset, limit=5)
        all_customers.extend(customers)
        
        if next_offset < 0:
            break
        offset = next_offset
    
    return all_customers


def get_customer(customer_id: str) -> Optional[CustomerFull]:
    """Get full customer details."""
    try:
        resp = _dev_api.dispatch(dev.Req_GetCustomer(id=customer_id))
        if resp.found and resp.company:
            c = resp.company
            return CustomerFull(
                id=c.id,
                name=c.name,
                brief=c.brief,
                location=c.location,
                deal_phase=c.deal_phase,
                high_level_status=c.high_level_status,
                account_manager=c.account_manager,
                primary_contact_name=c.primary_contact_name,
                primary_contact_email=c.primary_contact_email
            )
    except:
        pass
    return None


def search_customers(query: str = None, deal_phase: list = None, account_managers: list = None,
                     locations: list = None, offset: int = 0, limit: int = 5) -> tuple[list[CustomerBrief], int]:
    """Search customers with filters."""
    kwargs = {"offset": offset, "limit": limit}
    if query:
        kwargs["query"] = query
    if deal_phase:
        kwargs["deal_phase"] = deal_phase
    if account_managers:
        kwargs["account_managers"] = account_managers
    if locations:
        kwargs["locations"] = locations
    
    resp = _dev_api.dispatch(dev.Req_SearchCustomers(**kwargs))
    
    customers = []
    if resp.companies:
        for c in resp.companies:
            customers.append(CustomerBrief(
                id=c.id,
                name=c.name,
                location=c.location,
                deal_phase=c.deal_phase,
                high_level_status=c.high_level_status
            ))
    
    next_offset = resp.next_offset if resp.next_offset is not None else -1
    return customers, next_offset


# ============================================================
# PROJECTS
# ============================================================

def list_projects(offset: int = 0, limit: int = 5) -> tuple[list[ProjectBrief], int]:
    """Get a page of projects."""
    resp = _dev_api.dispatch(dev.Req_ListProjects(offset=offset, limit=limit))
    
    projects = []
    if resp.projects:
        for p in resp.projects:
            projects.append(ProjectBrief(
                id=p.id,
                name=p.name,
                customer=p.customer,
                status=p.status
            ))
    
    next_offset = resp.next_offset if resp.next_offset is not None else -1
    return projects, next_offset


def list_all_projects() -> list[ProjectBrief]:
    """Get ALL projects."""
    all_projects = []
    offset = 0
    
    while True:
        projects, next_offset = list_projects(offset=offset, limit=5)
        all_projects.extend(projects)
        
        if next_offset < 0:
            break
        offset = next_offset
    
    return all_projects


def get_project(project_id: str) -> Optional[ProjectFull]:
    """Get full project details."""
    try:
        resp = _dev_api.dispatch(dev.Req_GetProject(id=project_id))
        if resp.found and resp.project:
            p = resp.project
            team = []
            if p.team:
                for m in p.team:
                    team.append(TeamMember(
                        employee=m.employee,
                        time_slice=m.time_slice,
                        role=m.role
                    ))
            return ProjectFull(
                id=p.id,
                name=p.name,
                description=p.description,
                customer=p.customer,
                status=p.status,
                team=team
            )
    except:
        pass
    return None


def search_projects(query: str = None, customer_id: str = None, status: list = None,
                    team_employee: str = None, team_role: str = None,
                    include_archived: bool = False, offset: int = 0, limit: int = 5) -> tuple[list[ProjectBrief], int]:
    """Search projects with filters."""
    kwargs = {"offset": offset, "limit": limit, "include_archived": include_archived}
    if query:
        kwargs["query"] = query
    if customer_id:
        kwargs["customer_id"] = customer_id
    if status:
        kwargs["status"] = status
    if team_employee or team_role:
        team_filter = {}
        if team_employee:
            team_filter["employee_id"] = team_employee
        if team_role:
            team_filter["role"] = team_role
        kwargs["team"] = dev.ProjectTeamFilter(**team_filter)
    
    resp = _dev_api.dispatch(dev.Req_SearchProjects(**kwargs))
    
    projects = []
    if resp.projects:
        for p in resp.projects:
            projects.append(ProjectBrief(
                id=p.id,
                name=p.name,
                customer=p.customer,
                status=p.status
            ))
    
    next_offset = resp.next_offset if resp.next_offset is not None else -1
    return projects, next_offset


# ============================================================
# TIME ENTRIES
# ============================================================

def search_time_entries(employee: str = None, customer: str = None, project: str = None,
                        date_from: str = None, date_to: str = None, work_category: str = None,
                        billable: str = "", status: str = "",
                        offset: int = 0, limit: int = 5) -> tuple[list[TimeEntry], int, dict]:
    """
    Search time entries.
    
    Args:
        billable: "", "billable", or "non_billable"
        status: "", "draft", "submitted", "approved", "invoiced", "voided"
        
    Returns:
        Tuple of (entries, next_offset, totals)
        totals = {total_hours, total_billable, total_non_billable}
    """
    kwargs = {"offset": offset, "limit": limit}
    if employee:
        kwargs["employee"] = employee
    if customer:
        kwargs["customer"] = customer
    if project:
        kwargs["project"] = project
    if date_from:
        kwargs["date_from"] = date_from
    if date_to:
        kwargs["date_to"] = date_to
    if work_category:
        kwargs["work_category"] = work_category
    if billable:
        kwargs["billable"] = billable
    if status:
        kwargs["status"] = status
    
    resp = _dev_api.dispatch(dev.Req_SearchTimeEntries(**kwargs))
    
    entries = []
    if resp.entries:
        for e in resp.entries:
            entries.append(TimeEntry(
                id=e.id,
                employee=e.employee,
                customer=e.customer,
                project=e.project,
                date=e.date,
                hours=e.hours,
                work_category=e.work_category,
                notes=e.notes,
                billable=e.billable,
                status=e.status
            ))
    
    next_offset = resp.next_offset if resp.next_offset is not None else -1
    totals = {
        "total_hours": resp.total_hours,
        "total_billable": resp.total_billable,
        "total_non_billable": resp.total_non_billable
    }
    return entries, next_offset, totals


def get_time_entry(entry_id: str) -> Optional[TimeEntry]:
    """Get time entry by ID."""
    try:
        resp = _dev_api.dispatch(dev.Req_GetTimeEntry(id=entry_id))
        if resp.entry:
            e = resp.entry
            return TimeEntry(
                id=entry_id,  # entry doesn't have id field
                employee=e.employee,
                customer=e.customer,
                project=e.project,
                date=e.date,
                hours=e.hours,
                work_category=e.work_category,
                notes=e.notes,
                billable=e.billable,
                status=e.status
            )
    except:
        pass
    return None


class TimeSummaryByProject(BaseModel):
    """Time summary grouped by project."""
    customer: str
    project: str
    total_hours: float
    billable_hours: float
    non_billable_hours: float
    distinct_employees: int


class TimeSummaryByEmployee(BaseModel):
    """Time summary grouped by employee."""
    employee: str
    total_hours: float
    billable_hours: float
    non_billable_hours: float


def time_summary_by_project(date_from: str, date_to: str, 
                            customers: list = None, projects: list = None,
                            employees: list = None, billable: str = "") -> list[TimeSummaryByProject]:
    """
    Get time summaries grouped by project.
    
    Args:
        date_from: Start date (YYYY-MM-DD)
        date_to: End date (YYYY-MM-DD)
        customers: Optional list of customer IDs to filter
        projects: Optional list of project IDs to filter
        employees: Optional list of employee IDs to filter
        billable: "" | "billable" | "non_billable"
        
    Returns:
        List of TimeSummaryByProject objects
    """
    kwargs = {"date_from": date_from, "date_to": date_to}
    if customers:
        kwargs["customers"] = customers
    if projects:
        kwargs["projects"] = projects
    if employees:
        kwargs["employees"] = employees
    if billable:
        kwargs["billable"] = billable
    
    resp = _dev_api.dispatch(dev.Req_TimeSummaryByProject(**kwargs))
    
    summaries = []
    if resp.summaries:
        for s in resp.summaries:
            summaries.append(TimeSummaryByProject(
                customer=s.customer,
                project=s.project,
                total_hours=s.total_hours,
                billable_hours=s.billable_hours,
                non_billable_hours=s.non_billable_hours,
                distinct_employees=s.distinct_employees
            ))
    return summaries


def time_summary_by_employee(date_from: str, date_to: str,
                              customers: list = None, projects: list = None,
                              employees: list = None, billable: str = "") -> list[TimeSummaryByEmployee]:
    """
    Get time summaries grouped by employee.
    
    Args:
        date_from: Start date (YYYY-MM-DD)
        date_to: End date (YYYY-MM-DD)
        customers: Optional list of customer IDs to filter
        projects: Optional list of project IDs to filter
        employees: Optional list of employee IDs to filter
        billable: "" | "billable" | "non_billable"
        
    Returns:
        List of TimeSummaryByEmployee objects
    """
    kwargs = {"date_from": date_from, "date_to": date_to}
    if customers:
        kwargs["customers"] = customers
    if projects:
        kwargs["projects"] = projects
    if employees:
        kwargs["employees"] = employees
    if billable:
        kwargs["billable"] = billable
    
    resp = _dev_api.dispatch(dev.Req_TimeSummaryByEmployee(**kwargs))
    
    summaries = []
    if resp.summaries:
        for s in resp.summaries:
            summaries.append(TimeSummaryByEmployee(
                employee=s.employee,
                total_hours=s.total_hours,
                billable_hours=s.billable_hours,
                non_billable_hours=s.non_billable_hours
            ))
    return summaries


def log_time_entry(employee: str, date: str, hours: float, work_category: str,
                   notes: str, billable: bool, status: str, logged_by: str,
                   customer: str = None, project: str = None) -> str:
    """
    Log a new time entry.
    
    Returns:
        The created entry ID
    """
    kwargs = {
        "employee": employee,
        "date": date,
        "hours": hours,
        "work_category": work_category,
        "notes": notes,
        "billable": billable,
        "status": status,
        "logged_by": logged_by
    }
    if customer:
        kwargs["customer"] = customer
    if project:
        kwargs["project"] = project
    
    resp = _dev_api.dispatch(dev.Req_LogTimeEntry(**kwargs))
    print(f"\n{'='*60}")
    print(f"⚠️  STATE CHANGED - log_time_entry COMPLETED!")
    print(f"    Entry ID: {resp.id}")
    print(f"    Employee: {employee}, Project: {project}")
    print(f"    Hours: {hours}, Date: {date}, Billable: {billable}")
    print(f"    DO NOT call this function again - entry is already created!")
    print(f"{'='*60}\n")
    return resp.id


def update_time_entry(entry_id: str, date: str, hours: float, work_category: str,
                      notes: str, billable: bool, status: str, changed_by: str) -> bool:
    """
    Update existing time entry.
    NOTE: ALL fields must be provided (not partial update).
    """
    _dev_api.dispatch(dev.Req_UpdateTimeEntry(
        id=entry_id, date=date, hours=hours, work_category=work_category,
        notes=notes, billable=billable, status=status, changed_by=changed_by
    ))
    print(f"\n{'='*60}")
    print(f"⚠️  STATE CHANGED - update_time_entry COMPLETED!")
    print(f"    Entry ID: {entry_id}")
    print(f"    Hours: {hours}, Date: {date}, Status: {status}")
    print(f"    DO NOT call this function again - entry is already updated!")
    print(f"{'='*60}\n")
    return True


# ============================================================
# WIKI
# ============================================================

def list_wiki() -> list[str]:
    """List all wiki file paths."""
    resp = _dev_api.dispatch(dev.Req_ListWiki())
    return resp.paths or []


def load_wiki(file_path: str) -> Optional[str]:
    """Load wiki file content."""
    try:
        resp = _dev_api.dispatch(dev.Req_LoadWiki(file=file_path))
        return resp.content
    except:
        return None


def search_wiki(query_regex: str) -> list[dict]:
    """
    Search wiki with regex.
    
    Returns:
        List of {path, linum, content} dicts
    """
    resp = _dev_api.dispatch(dev.Req_SearchWiki(query_regex=query_regex))
    results = []
    if resp.results:
        for r in resp.results:
            results.append({
                "path": r.path,
                "linum": r.linum,
                "content": r.content
            })
    return results


# ============================================================
# FUZZY MATCHING (General Purpose)
# ============================================================

def fuzzy_compare(targets: list[str], candidates: list[str], top_n: int = 5) -> list[dict]:
    """
    Compare two lists of words/strings using fuzzy matching.
    Always returns top N matches - agent decides how to interpret results.
    
    Args:
        targets: List of target words to find matches for
        candidates: List of candidate words to search in
        top_n: Return top N results per target, default 5
        
    Returns:
        List of {target, candidate, ratio} dicts, sorted by ratio descending
        
    Example:
        fuzzy_compare(["felix"], ["Felix Baum", "jane_doe", "john_smith"])
        → [{"target": "felix", "candidate": "Felix Baum", "ratio": 0.67}, ...]
    """
    from difflib import SequenceMatcher
    
    results = []
    for target in targets:
        target_lower = target.lower()
        target_results = []
        for candidate in candidates:
            candidate_lower = candidate.lower()
            ratio = SequenceMatcher(None, target_lower, candidate_lower).ratio()
            target_results.append({
                "target": target,
                "candidate": candidate,
                "ratio": round(ratio, 3)
            })
        # Sort this target's results and take top N
        target_results.sort(key=lambda x: x["ratio"], reverse=True)
        results.extend(target_results[:top_n])
    
    # Final sort by ratio descending
    results.sort(key=lambda x: x["ratio"], reverse=True)
    return results


def fuzzy_find_in_text(targets: list[str], texts: list[str], top_n: int = 10, context_chars: int = 100) -> list[dict]:
    """
    Find target words/phrases inside larger texts using sliding window fuzzy matching.
    Always returns top N matches - agent decides how to interpret results.
    
    Args:
        targets: List of words/phrases to search for
        texts: List of texts to search in (can be large documents)
        top_n: Return top N results per target, default 10
        context_chars: Characters of context around match, default 100
        
    Returns:
        List of {target, text_index, matched, ratio, position, context} dicts
        
    Example:
        fuzzy_find_in_text(["employee"], ["The employe works here", "Other text"])
        → [{"target": "employee", "text_index": 0, "matched": "employe", "ratio": 0.93, ...}]
    """
    from difflib import SequenceMatcher
    
    results = []
    for target in targets:
        target_lower = target.lower()
        target_len = len(target_lower)
        target_results = []
        
        for text_idx, text in enumerate(texts):
            if not text:
                continue
            
            text_lower = text.lower()
            
            # Sliding window scan
            for i in range(len(text_lower) - target_len + 1):
                window = text_lower[i:i + target_len]
                
                # Quick pre-check: first char should match for speed
                if window[0] != target_lower[0]:
                    continue
                
                ratio = SequenceMatcher(None, target_lower, window).ratio()
                
                # Get context
                start = max(0, i - context_chars)
                end = min(len(text), i + target_len + context_chars)
                context = text[start:end].strip()
                
                target_results.append({
                    "target": target,
                    "text_index": text_idx,
                    "matched": text[i:i + target_len],
                    "ratio": round(ratio, 3),
                    "position": i,
                    "context": context
                })
        
        # Sort this target's results and take top N
        target_results.sort(key=lambda x: x["ratio"], reverse=True)
        results.extend(target_results[:top_n])
    
    # Final sort by ratio descending
    results.sort(key=lambda x: x["ratio"], reverse=True)
    return results


def search_wiki_fuzzy(words: list[str], top_n: int = 10, context_chars: int = 100) -> list[dict]:
    """
    Fuzzy search wiki for words/phrases. Uses fuzzy_find_in_text internally.
    Always returns top N matches - agent decides how to interpret results.
    
    Args:
        words: List of words or phrases to search for
        top_n: Return top N results per word, default 10
        context_chars: Characters of context around match, default 100
        
    Returns:
        List of {path, word, matched, ratio, line_num, context} dicts
    """
    results = []
    paths = list_wiki()
    
    # Collect all wiki content
    wiki_contents = []
    wiki_paths = []
    for path in paths:
        content = load_wiki(path)
        if content:
            wiki_contents.append(content)
            wiki_paths.append(path)
    
    # Use fuzzy_find_in_text with high top_n to get candidates from all wikis
    matches = fuzzy_find_in_text(words, wiki_contents, top_n=top_n * len(wiki_paths), context_chars=context_chars)
    
    for match in matches:
        path = wiki_paths[match["text_index"]]
        content = wiki_contents[match["text_index"]]
        line_num = content[:match["position"]].count('\n') + 1
        results.append({
            "path": path,
            "word": match["target"],
            "matched": match["matched"],
            "ratio": match["ratio"],
            "line_num": line_num,
            "context": match["context"]
        })
    
    # Sort and limit to top_n per word
    results.sort(key=lambda x: x["ratio"], reverse=True)
    return results[:top_n * len(words)]


def update_wiki(file_path: str, content: str, changed_by: str = None) -> bool:
    """Update wiki file content."""
    kwargs = {"file": file_path, "content": content}
    if changed_by:
        kwargs["changed_by"] = changed_by
    _dev_api.dispatch(dev.Req_UpdateWiki(**kwargs))
    print(f"[UPDATE_WIKI] Updated: file={file_path}, content_length={len(content)}, changed_by={changed_by}")
    return True


# ============================================================
# UPDATES
# ============================================================

def update_employee_info(employee_id: str, notes: str = None, salary: int = None,
                         skills: list = None, wills: list = None,
                         location: str = None, department: str = None,
                         changed_by: str = None) -> Optional[EmployeeFull]:
    """Update employee information."""
    kwargs = {"employee": employee_id}
    if notes is not None:
        kwargs["notes"] = notes
    if salary is not None:
        kwargs["salary"] = salary
    if skills is not None:
        kwargs["skills"] = [dev.SkillLevel(**s) for s in skills]
    if wills is not None:
        kwargs["wills"] = [dev.SkillLevel(**w) for w in wills]
    if location is not None:
        kwargs["location"] = location
    if department is not None:
        kwargs["department"] = department
    if changed_by is not None:
        kwargs["changed_by"] = changed_by
    
    resp = _dev_api.dispatch(dev.Req_UpdateEmployeeInfo(**kwargs))
    # Build fields string for logging
    fields = {k: v for k, v in [('salary', salary), ('notes', notes), ('location', location), ('department', department)] if v is not None}
    print(f"\n{'='*60}")
    print(f"⚠️  STATE CHANGED - update_employee_info COMPLETED!")
    print(f"    Employee: {employee_id}")
    print(f"    Updated fields: {fields}")
    print(f"    DO NOT call this function again - state is already modified!")
    print(f"{'='*60}\n")
    if resp.employee:
        e = resp.employee
        return EmployeeFull(
            id=e.id,
            name=e.name,
            email=e.email,
            salary=e.salary,
            notes=e.notes,
            location=e.location,
            department=e.department,
            skills=[SkillLevel(name=s.name, level=s.level) for s in (e.skills or [])],
            wills=[SkillLevel(name=w.name, level=w.level) for w in (e.wills or [])]
        )
    return None


def update_project_status(project_id: str, status: str, changed_by: str = None) -> bool:
    """Update project status (idea, exploring, active, paused, archived)."""
    kwargs = {"id": project_id, "status": status}
    if changed_by:
        kwargs["changed_by"] = changed_by
    _dev_api.dispatch(dev.Req_UpdateProjectStatus(**kwargs))
    print(f"\n{'='*60}")
    print(f"⚠️  STATE CHANGED - update_project_status COMPLETED!")
    print(f"    Project: {project_id}")
    print(f"    New Status: {status}")
    print(f"    DO NOT call this function again - status is already changed!")
    print(f"{'='*60}\n")
    return True


def update_project_team(project_id: str, team: list, changed_by: str = None) -> bool:
    """
    Update project team.
    
    Args:
        project_id: Project ID
        team: List of TeamMember objects OR dicts with {employee, time_slice, role}
        changed_by: User making the change
    """
    kwargs = {"id": project_id}
    if team:
        # Accept both TeamMember objects and dicts
        workloads = []
        for t in team:
            if isinstance(t, dict):
                workloads.append(dev.Workload(
                    employee=t["employee"],
                    time_slice=t["time_slice"],
                    role=t["role"]
                ))
            else:
                # TeamMember object
                workloads.append(dev.Workload(
                    employee=t.employee,
                    time_slice=t.time_slice,
                    role=t.role
                ))
        kwargs["team"] = workloads
    if changed_by:
        kwargs["changed_by"] = changed_by
    _dev_api.dispatch(dev.Req_UpdateProjectTeam(**kwargs))
    # Build summary from either dict or object
    team_summary = []
    for t in (team or []):
        if isinstance(t, dict):
            team_summary.append(f"{t['employee']}({t['role']})")
        else:
            team_summary.append(f"{t.employee}({t.role})")
    print(f"\n{'='*60}")
    print(f"⚠️  STATE CHANGED - update_project_team COMPLETED!")
    print(f"    Project: {project_id}")
    print(f"    Team: {team_summary}")
    print(f"    DO NOT call this function again - team is already updated!")
    print(f"{'='*60}\n")
    return True


# ============================================================
# Initialization helper for executor
# ============================================================

def _init_dev_api(dev_api_client, dev_module):
    """Initialize dev API (called by executor)."""
    global _dev_api
    _dev_api = dev_api_client
    # Note: dev module is already imported at top level


# ============================================================
# Export all functions for easy import
# ============================================================

__all__ = [
    # Init
    '_init_dev_api', 'init_dev', 'get_core', 'get_api',
    # Types
    'SkillLevel', 'EmployeeBrief', 'EmployeeFull', 'CustomerBrief', 'CustomerFull',
    'ProjectBrief', 'ProjectFull', 'TeamMember', 'TimeEntry', 'WhoAmI',
    'TimeSummaryByProject', 'TimeSummaryByEmployee',
    # Who am I
    'who_am_i',
    # Employees
    'list_employees', 'list_all_employees', 'get_employee', 'search_employees',
    # Customers
    'list_customers', 'list_all_customers', 'get_customer', 'search_customers',
    # Projects
    'list_projects', 'list_all_projects', 'get_project', 'search_projects',
    # Time entries
    'search_time_entries', 'get_time_entry', 'log_time_entry', 'update_time_entry',
    'time_summary_by_project', 'time_summary_by_employee',
    # Wiki
    'list_wiki', 'load_wiki', 'search_wiki', 'search_wiki_fuzzy', 'update_wiki',
    # Fuzzy matching (general purpose)
    'fuzzy_compare', 'fuzzy_find_in_text',
    # Updates
    'update_employee_info', 'update_project_status', 'update_project_team',
    # NOTE: provide_response is NOT exported - it's called separately in runner
    # NOTE: delegate_to_agent REMOVED - linear agent only
]
