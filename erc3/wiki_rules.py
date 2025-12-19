"""
Wiki Rules Distillation Module.

Loads wiki pages, distills them into compact RFC-style rules using LLM,
and caches the results based on wiki SHA1 hash.

The distilled rules are dynamically injected into all prompts.

Thread-safe for multiprocessing: uses file-based locking for cache writes.
"""

import json
import os
import fcntl
import time
from pathlib import Path
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

from .utils import llm_structured, LLM_MODEL_PLAN


# ============================================================
# Data Models
# ============================================================

RuleCategory = Literal["applies_to_guests", "applies_to_users", "other"]


class Rule(BaseModel):
    """A distilled rule from wiki content."""
    why_relevant_summary: str = Field(..., description="Why this rule is relevant")
    category: RuleCategory = Field(..., description="Who this rule applies to")
    compact_rule: str = Field(..., description="Compact RFC-style rule text")


class DistilledWikiRules(BaseModel):
    """Distilled wiki content."""
    company_name: str = Field(..., description="Name of the company")
    company_locations: List[str] = Field(default_factory=list, description="Locations where company operates")
    company_execs: List[str] = Field(default_factory=list, description="Executive names")
    rules: List[Rule] = Field(default_factory=list, description="Distilled rules")


class WikiRulesContext(BaseModel):
    """Context containing distilled wiki rules for injection into prompts."""
    wiki_sha1: str
    distilled: DistilledWikiRules
    is_public: bool = False
    current_user: Optional[str] = None
    current_user_info: Optional[str] = None  # JSON dump of employee info
    today: Optional[str] = None
    
    def get_formatted_rules(self) -> str:
        """Format rules for inclusion in prompts, grouped by category."""
        lines = []
        
        # Company context
        lines.append(f"# Company: {self.distilled.company_name}")
        if self.distilled.company_locations:
            lines.append(f"Locations: {', '.join(self.distilled.company_locations)}")
        if self.distilled.company_execs:
            lines.append(f"Executives: {', '.join(self.distilled.company_execs)}")
        
        # Group rules by category
        lines.append(
            "\n## Company rules (you should strictly follow them)\n"
            "**IF NESSESARY TAKE ACTIONS TO COMPLY WITH THE COMPANY RULES BELOW**"
        )
        
        # Determine which categories to show based on user type
        if self.is_public:
            categories_to_show = [
                ("applies_to_guests", "Guest/Public User Rules"),
                ("other", "General Rules"),
            ]
        else:
            categories_to_show = [
                ("applies_to_users", "Authenticated User Rules"),
                ("other", "General Rules"),
            ]
        
        for category_key, category_label in categories_to_show:
            category_rules = [r for r in self.distilled.rules if r.category == category_key]
            if category_rules:
                lines.append(f"\n## {category_label}:")
                for rule in category_rules:
                    lines.append(f"- {rule.compact_rule}")
        
        # Current context
        lines.append(f"\n# Current Context")
        lines.append(f"Date: {self.today}")
        
        if self.is_public:
            lines.append("Current actor: GUEST (Anonymous/Public user)")
        else:
            lines.append(f"Current actor: Authenticated user {self.current_user}")
            if self.current_user_info:
                lines.append(f"User details: {self.current_user_info}")
        
        return "\n".join(lines)


# ============================================================
# Cache Directory
# ============================================================

def _get_cache_dir() -> Path:
    """Get cache directory for wiki rules."""
    cache_dir = Path(os.path.dirname(__file__)) / "wiki_cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def _get_cache_path(wiki_sha1: str) -> Path:
    """Get cache file path for a wiki SHA1."""
    return _get_cache_dir() / f"context_{wiki_sha1}_v2.json"


# ============================================================
# Distillation Prompt
# ============================================================

DISTILL_WIKI_PROMPT = """Carefully review the wiki content below and identify the most important security, scoping, and data rules that will be highly relevant for an AI agent automating company APIs.

Pay attention to rules that mention:
- AI Agent / Public ChatBot permissions
- Access control and security policies
- Data privacy and confidentiality
- Role-based permissions
- Guest vs authenticated user restrictions

Rules must be compact RFC-style, using pseudo-code for compactness where appropriate.

Categories:
- applies_to_guests: Rules for public/anonymous users (ChatBot users)
- applies_to_users: Rules for authenticated employees
- other: General rules that apply to everyone

## Wiki Content:
{wiki_content}

Extract:
1. Company name
2. Company locations
3. Executive names
4. All relevant security/access/data rules
"""


# ============================================================
# Main Functions
# ============================================================

def distill_rules_from_wiki(dev_api, wiki_sha1: str) -> DistilledWikiRules:
    """
    Distill wiki content into compact rules.
    
    Thread-safe: uses file-based locking for concurrent cache access.
    Multiple processes can read the cache, but only one can write at a time.
    
    Args:
        dev_api: ERC3 dev client for accessing wiki
        wiki_sha1: Current wiki SHA1 for caching
        
    Returns:
        DistilledWikiRules with company info and rules
    """
    cache_path = _get_cache_path(wiki_sha1)
    lock_path = cache_path.with_suffix(".lock")
    
    # Check cache first (before acquiring lock)
    if cache_path.exists():
        try:
            return DistilledWikiRules.model_validate_json(cache_path.read_text())
        except Exception:
            pass  # Cache invalid, re-distill
    
    # Acquire file lock for writing
    # This prevents multiple processes from distilling the same wiki simultaneously
    lock_file = open(lock_path, "w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        
        # Double-check: another process might have created cache while we waited
        if cache_path.exists():
            try:
                result = DistilledWikiRules.model_validate_json(cache_path.read_text())
                return result
            except Exception:
                pass  # Cache invalid, re-distill
        
        # Load wiki content
        pid = os.getpid()
        print(f"[PID {pid}] Distilling wiki rules (sha1={wiki_sha1[:8]}...)...")
        
        wiki_content = ""
        try:
            from erc3 import erc3 as dev
            paths_resp = dev_api.dispatch(dev.Req_ListWiki())
            paths = paths_resp.paths or []
            
            for path in paths:
                content_resp = dev_api.dispatch(dev.Req_LoadWiki(file=path))
                if content_resp.content:
                    wiki_content += f"\n---- {path} ----\n{content_resp.content}\n"
        except Exception as e:
            print(f"Warning: Could not load wiki: {e}")
            # Return empty rules if wiki not accessible
            return DistilledWikiRules(
                company_name="Unknown",
                company_locations=[],
                company_execs=[],
                rules=[]
            )
        
        if not wiki_content.strip():
            return DistilledWikiRules(
                company_name="Unknown",
                company_locations=[],
                company_execs=[],
                rules=[]
            )
        
        # Distill using LLM
        prompt = DISTILL_WIKI_PROMPT.format(wiki_content=wiki_content)
        distilled = llm_structured(prompt, DistilledWikiRules, model=LLM_MODEL_PLAN)
        
        # Cache result (atomic write using temp file)
        temp_path = cache_path.with_suffix(".tmp")
        temp_path.write_text(distilled.model_dump_json(indent=2))
        temp_path.rename(cache_path)  # Atomic on POSIX
        
        print(f"[PID {pid}] Wiki rules distilled and cached: {len(distilled.rules)} rules")
        
        return distilled
        
    finally:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()
        # Clean up lock file (best effort)
        try:
            lock_path.unlink()
        except Exception:
            pass


def create_wiki_rules_context(dev_api, user_context: dict) -> WikiRulesContext:
    """
    Create wiki rules context for a task.
    
    Args:
        dev_api: ERC3 dev client
        user_context: User context from who_am_i() with wiki_sha1
        
    Returns:
        WikiRulesContext ready for injection into prompts
    """
    wiki_sha1 = user_context.get("wiki_sha1", "")
    
    # Distill rules (uses cache if available)
    distilled = distill_rules_from_wiki(dev_api, wiki_sha1)
    
    # Get current user info if authenticated
    current_user_info = None
    if not user_context.get("is_public") and user_context.get("current_user"):
        try:
            from erc3 import erc3 as dev
            emp_resp = dev_api.dispatch(dev.Req_GetEmployee(id=user_context["current_user"]))
            if emp_resp.employee:
                # Serialize employee info (excluding sensitive fields for compact prompt)
                emp = emp_resp.employee
                emp_data = {
                    "id": emp.id,
                    "name": emp.name,
                    "department": emp.department,
                    "location": emp.location
                }
                current_user_info = json.dumps(emp_data)
        except Exception:
            pass
    
    return WikiRulesContext(
        wiki_sha1=wiki_sha1,
        distilled=distilled,
        is_public=user_context.get("is_public", False),
        current_user=user_context.get("current_user"),
        current_user_info=current_user_info,
        today=user_context.get("today")
    )


# ============================================================
# Default empty context (for when wiki is not available)
# ============================================================

def get_empty_wiki_rules() -> str:
    """Get empty wiki rules placeholder."""
    return "# Wiki Rules: Not available"
