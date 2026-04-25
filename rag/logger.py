# logger.py
# Author: Frank Afelete Kofi Dogli | Index: 10022200146
# Part D: Pipeline Stage Logger

import json
import os
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

_session_log = []


def log_stage(stage, data):
    """Log a pipeline stage with timestamp."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "stage": stage,
        "data": data,
    }
    _session_log.append(entry)

    # Write to file as well
    log_path = os.path.join(LOG_DIR, "pipeline_log.jsonl")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return entry


def get_session_log():
    """Return all log entries from this session."""
    return _session_log


def clear_session_log():
    """Clear session log between queries."""
    _session_log.clear()


def format_log_for_display(log_entries):
    """Format log entries for display in Streamlit."""
    lines = []
    for e in log_entries:
        lines.append(f"**[{e['stage'].upper()}]** `{e['timestamp']}`")
        for k, v in e["data"].items():
            if isinstance(v, list):
                lines.append(f"  - **{k}**: {len(v)} items")
            elif isinstance(v, str) and len(v) > 200:
                lines.append(f"  - **{k}**: {v[:200]}...")
            else:
                lines.append(f"  - **{k}**: {v}")
        lines.append("")
    return "\n".join(lines)