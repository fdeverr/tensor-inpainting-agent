"""Registry construction kept separate from workflow orchestration."""

from .framework import ToolRegistry

from .research_tools import (
    AnalyzeImageTool,
    CompareExperimentsTool,
    EvaluateReconstructionTool,
    RunInterpolationTool,
    TrainTensorModelTool,
    TuneTensorModelTool,
)


def build_research_tool_registry() -> ToolRegistry:
    """Register every deterministic Day 3 research tool."""

    registry = ToolRegistry()
    for tool in (
        AnalyzeImageTool(),
        RunInterpolationTool(),
        TuneTensorModelTool(),
        TrainTensorModelTool(),
        EvaluateReconstructionTool(),
        CompareExperimentsTool(),
    ):
        registry.register_tool(tool)
    return registry
