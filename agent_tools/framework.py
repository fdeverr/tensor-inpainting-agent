"""Import the sibling Hello Agents source tree in both supported launch modes.

The learning project is intentionally a standalone folder inside the
``hello_agents`` source checkout.  Users may launch it from that checkout as
``python -m inpainting_research_agent...`` or install/import the parent package.
"""

from __future__ import annotations

import sys
from pathlib import Path


try:
    import hello_agents  # noqa: F401
except ModuleNotFoundError as error:
    if error.name != "hello_agents":
        raise
    repository_parent = Path(__file__).resolve().parents[3]
    if str(repository_parent) not in sys.path:
        sys.path.insert(0, str(repository_parent))

from hello_agents.observability.trace_logger import TraceLogger
from hello_agents.tools.base import Tool, ToolParameter
from hello_agents.tools.errors import ToolErrorCode
from hello_agents.tools.registry import ToolRegistry
from hello_agents.tools.response import ToolResponse, ToolStatus

__all__ = [
    "Tool",
    "ToolParameter",
    "ToolErrorCode",
    "ToolRegistry",
    "ToolResponse",
    "ToolStatus",
    "TraceLogger",
]
