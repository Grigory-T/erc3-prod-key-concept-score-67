FROM python:3.12-slim

RUN addgroup --system app && adduser --system --ingroup app --disabled-password app

WORKDIR /opt/app

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://erc.timetoact-group.at/ -r requirements.txt

# Copy executor files only (no agent code - runs outside Docker)
COPY --chown=app:app executor.py dev_functions.py ./

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=3 \
    PYTHONNOUSERSITE=1

# ERC3_API_KEY and OPENROUTER_API_KEY passed at runtime via docker-compose

USER app:app

EXPOSE 8001

# Simple executor - no sub-agents, just code execution
# IMPORTANT: Use single uvicorn worker to ensure WORKERS dict is shared across requests
# Multiple workers would create separate WORKERS dicts, breaking global persistence
# Single worker can still handle concurrent async requests efficiently
CMD ["uvicorn", "executor:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1", "--no-access-log"]
