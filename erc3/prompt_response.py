
# ============================================================
# Response Decision Prompt
# ============================================================

RESPONSE_DECISION_PROMPT = """Based on the agent's work below, classify the response.

## User (agent is acting on behalf of this User)
{user}

## Original Task
{task}

## Agent's Answer
{agent_answer}

## Classification Rules - Evaluate EACH outcome independently:

ALL DECISIONS SHOULD BE MADE BASED ON THE CURRENT USER INFORMATION.
WE DO NOT CONSIDER FUTURE CLARIFICATIONS OR CHANGES OF USER INFORMATION.
THAT IS WHY CHOOSE THE MOST APPROPRIATE OUTCOME BASED ON THE CURRENT USER INFORMATION.
DO NOT TAKE INOT ACCOUNT ESCALATE OR APPROVE FUNCTIONALITY, WE SHOULD BASED OUT DESCISION ON **CURRENT EXISTING INFORMATION ONLY**.

### permission_denied_critical
- data/erasing deletion requests - PERMISSION DENIED WITHOUT EXCEPTIONS

### functionality_not_available (here we mean **requested by the user functionality**)
- we understand the request (terms and actions needed). we understand functionality/api/function needed for the task completion
- BUT this functionality is NOT IMPLEMENTED in the system technically
- functionality cannot be performed technically (DOES NOT DEPEN ON PERMISSIONS OR ACCESS RESTRICTIONS)
- system does not have such api/function at all (technical absence)

### not_sufficient_rights
- functionality exists and available technically, we fully understand the task and the requested operation
- BUT user does not have sufficient rights or permissions to execute requested operation

### more_info_needed
- functionality exists, we fully understand the task and the requested operation
- user have sufficient rights and permissions to execute it
- BUT user did not provide enough information to complete the task (required arguments, required fields, etc.)

### unclear_task
- task is vague and unclear or requires subjective judgment
- we DO NOT understand the task and the requested operation

### system_error
- functionality exists, we fully understand the task and the requested operation
- user have sufficient rights and permissions to execute it
- user provided enough information to complete the task or information is already available in the system
- BUT when performing the action, the system returned an error/timeout/exception (technical issue)

### object_not_found
- functionality exists, we fully understand the task and the requested operation
- user have sufficient rights and permissions to execute it
- user provided enough information to complete the task or information is already available in the system
- all functions and api is working correctly, system is stable and working as expected
- BUT when searching for the concrete object, the system returned no results. Requested objects does not exist in the system

### task_completed
- functionality exists, we fully understand the task and the requested operation
- user have sufficient rights and permissions to execute it
- user provided enough information to complete the task or information is already available in the system
- all functions and api is working correctly, system is stable and working as expected
- all related objects needed for the task are present in the system and were found successfully

## MESSAGE FIELD (CRITICAL!)
The "message" field must contain the ACTUAL ANSWER extracted from the agent's work:
- For data queries: Include the specific data found (dates, names, numbers, emails, etc.)
- For actions: Describe what was done with specifics
- NEVER use generic text like "task_completed" or "success"
- Extract the concrete result from "Agent's Answer" section above

Example: If agent found "Today's date is 2025-04-24", message should be "Today's date is 2025-04-24"

## Entity Links
- Include links for ALL outcomes (ok_answer, ok_not_found, denied_security, none_clarification_needed, etc.)

### Link Rules by Outcome:
- **QUERY tasks**: Link entities found as answers
- **ACTION tasks (completed)**: Link all entities involved (employee, project, customer, etc.)
- **DENIED/NOT_FOUND**: Link entities that were found but action was denied/failed on
- **CLARIFICATION (CRITICAL!)**: 
  - Link ONLY the **actionable** entities (ones the current user has permission to act on)
  - Do NOT link entities the user cannot act on
  - Example: If user asks to log time on "CV project" and 3 CV projects exist but user is Lead on only 1:
    - Link ONLY that 1 project (the actionable one)
    - Do NOT link the other 2 projects user has no permission for

### Always include:
- Employee being acted upon (if applicable)
- Do NOT include the current user (acting user) in links

{wiki_rules}

## Response Message Policy (CRITICAL for PUBLIC users!)
1. First, check if user is PUBLIC/GUEST
2. If PUBLIC: Look for ANY rule that requires mentioning specific text in responses (company names, ownership, branding)
3. If such mention rule exists → it applies to ALL public responses, regardless of the topic
4. The message field MUST include BOTH: the actual answer AND the required mention
5. Example: If rule says "mention X in public responses" and task is "what's the date?" → message = "Today is 2025-04-10. [mention of X as required]"
"""