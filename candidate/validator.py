"""AST policy checks plus a short independent-process model smoke test."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


ALLOWED_IMPORTS = {
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "math",
}
FORBIDDEN_MODULES = {
    "os",
    "subprocess",
    "socket",
    "requests",
    "shutil",
    "pathlib",
    "urllib",
    "http",
    "importlib",
}
FORBIDDEN_CALLS = {
    "open",
    "eval",
    "exec",
    "compile",
    "__import__",
    "input",
}
ALLOWED_BASE_NAMES = {
    "BaseTensorInpaintingModel",
    "MatrixFactorization",
    "CPDecomposition",
    "TuckerDecomposition",
}
REQUIRED_CLASS_NAME = "CandidateTensorInpaintingModel"


def _root_name(node: ast.AST) -> Optional[str]:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _static_checks(source: str) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    try:
        tree = ast.parse(source)
        checks.append({"name": "ast_parse", "passed": True})
    except SyntaxError as error:
        return {
            "passed": False,
            "checks": [{"name": "ast_parse", "passed": False}],
            "errors": [
                {
                    "code": "SYNTAX_ERROR",
                    "message": str(error),
                    "line": error.lineno,
                }
            ],
        }

    candidate_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_IMPORTS:
                    errors.append(
                        {
                            "code": "FORBIDDEN_IMPORT",
                            "message": "import %s is not allowed" % alias.name,
                            "line": node.lineno,
                        }
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level != 0 or module not in ALLOWED_IMPORTS:
                errors.append(
                    {
                        "code": "FORBIDDEN_IMPORT",
                        "message": "from %s import ... is not allowed" % module,
                        "line": node.lineno,
                    }
                )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                errors.append(
                    {
                        "code": "FORBIDDEN_CALL",
                        "message": "%s() is not allowed" % node.func.id,
                        "line": node.lineno,
                    }
                )
            root = _root_name(node.func)
            if root in FORBIDDEN_MODULES:
                errors.append(
                    {
                        "code": "FORBIDDEN_CALL",
                        "message": "calls through %s are not allowed" % root,
                        "line": node.lineno,
                    }
                )
        elif isinstance(node, ast.ClassDef) and node.name == REQUIRED_CLASS_NAME:
            candidate_class = node

    checks.append({"name": "import_and_call_policy", "passed": not errors})
    if candidate_class is None:
        errors.append(
            {
                "code": "MISSING_CLASS",
                "message": "required class %s was not found" % REQUIRED_CLASS_NAME,
            }
        )
    else:
        base_names = {
            base.id for base in candidate_class.bases if isinstance(base, ast.Name)
        }
        if not base_names.intersection(ALLOWED_BASE_NAMES):
            errors.append(
                {
                    "code": "INVALID_BASE_CLASS",
                    "message": "candidate must inherit the tensor-model interface or an approved baseline",
                    "line": candidate_class.lineno,
                }
            )
        search_methods = [
            node
            for node in candidate_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "search_space"
        ]
        if not search_methods:
            errors.append(
                {
                    "code": "MISSING_SEARCH_SPACE",
                    "message": "candidate must declare search_space()",
                    "line": candidate_class.lineno,
                }
            )
        elif not any(
            isinstance(decorator, ast.Name) and decorator.id == "classmethod"
            for decorator in search_methods[0].decorator_list
        ):
            errors.append(
                {
                    "code": "INVALID_SEARCH_SPACE",
                    "message": "search_space() must be a classmethod",
                    "line": search_methods[0].lineno,
                }
            )
    checks.append(
        {
            "name": "candidate_contract",
            "passed": not any(
                error["code"]
                in {"MISSING_CLASS", "INVALID_BASE_CLASS", "MISSING_SEARCH_SPACE", "INVALID_SEARCH_SPACE"}
                for error in errors
            ),
        }
    )
    return {"passed": not errors, "checks": checks, "errors": errors}


class CandidateValidator:
    """Validate generated code without importing it into the controller process."""

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = float(timeout_seconds)

    def validate(
        self,
        model_path: str,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        started_at = time.perf_counter()
        path = Path(model_path)
        if not path.is_file():
            raise ValueError("candidate model does not exist: %s" % path)
        source = path.read_text(encoding="utf-8")
        code_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        static = _static_checks(source)
        smoke: Dict[str, Any] = {
            "passed": False,
            "skipped": True,
            "reason": "static validation failed",
        }
        if static["passed"]:
            runner = Path(__file__).resolve().with_name("smoke_runner.py")
            environment = dict(os.environ)
            for name in list(environment):
                upper_name = name.upper()
                if "API_KEY" in upper_name or "TOKEN" in upper_name or "SECRET" in upper_name:
                    environment.pop(name, None)
            environment["CUDA_VISIBLE_DEVICES"] = ""
            environment["PYTHONNOUSERSITE"] = "1"
            try:
                completed = subprocess.run(
                    [sys.executable, "-I", str(runner), str(path)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                    shell=False,
                    env=environment,
                )
                output_lines = [
                    line for line in completed.stdout.splitlines() if line.strip()
                ]
                parsed = json.loads(output_lines[-1]) if output_lines else {}
                smoke = {
                    **parsed,
                    "skipped": False,
                    "return_code": completed.returncode,
                    "stderr": completed.stderr[-4000:],
                }
                smoke["passed"] = bool(parsed.get("passed")) and completed.returncode == 0
            except subprocess.TimeoutExpired:
                smoke = {
                    "passed": False,
                    "skipped": False,
                    "error_type": "TimeoutExpired",
                    "message": "smoke test exceeded %.2f seconds" % self.timeout_seconds,
                }
            except (json.JSONDecodeError, OSError) as error:
                smoke = {
                    "passed": False,
                    "skipped": False,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }

        passed = bool(static["passed"] and smoke["passed"])
        feedback = []
        feedback.extend(error["message"] for error in static["errors"])
        if not smoke["passed"] and not smoke.get("skipped"):
            feedback.append(smoke.get("message", "candidate smoke test failed"))
        result = {
            "passed": passed,
            "status": "validated" if passed else "rejected",
            "code_sha256": code_hash,
            "static_validation": static,
            "smoke_test": smoke,
            "feedback": feedback,
            "runtime_seconds": float(time.perf_counter() - started_at),
            "safety_note": (
                "AST checks and an isolated process are MVP guardrails, not a "
                "production-grade security sandbox."
            ),
        }
        if output_path is not None:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
        return result
