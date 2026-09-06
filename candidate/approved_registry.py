"""Immutable, auditable promotion of candidates that pass the experiment judge."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def promote_candidate(
    candidate_dir: str,
    approved_root: str,
    algorithm_name: str,
    source_run_id: str,
    best_config: Dict[str, Any],
    comparison: Dict[str, Any],
    conditions: Dict[str, Any],
) -> Dict[str, Any]:
    """Copy a validated, accepted candidate into a versioned approved directory."""

    if comparison.get("accepted") is not True:
        raise ValueError("only an accepted candidate can be promoted")
    source = Path(candidate_dir)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    validation = json.loads((source / "validation.json").read_text(encoding="utf-8"))
    if manifest.get("eligible_for_training") is not True or validation.get("passed") is not True:
        raise ValueError("candidate did not pass its code validation gate")
    code = (source / "model.py").read_text(encoding="utf-8")
    code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
    if code_hash != manifest.get("code_sha256"):
        raise ValueError("candidate code changed after validation")

    algorithm_root = Path(approved_root) / algorithm_name
    algorithm_root.mkdir(parents=True, exist_ok=True)
    existing_versions = [
        int(path.name[1:])
        for path in algorithm_root.glob("v[0-9]*")
        if path.is_dir() and path.name[1:].isdigit()
    ]
    version = "v%d" % (max(existing_versions, default=0) + 1)
    destination = algorithm_root / version
    destination.mkdir(parents=False, exist_ok=False)
    for filename in ("model.py", "idea.json", "manifest.json", "validation.json"):
        shutil.copy2(source / filename, destination / filename)
    _write_json(destination / "best_config.json", best_config)
    _write_json(destination / "comparison.json", comparison)
    approved_manifest = {
        "algorithm_name": algorithm_name,
        "version": version,
        "source_candidate_id": manifest["candidate_id"],
        "source_run_id": source_run_id,
        "base_method": manifest["base_method"],
        "code_sha256": code_hash,
        "promoted_at": datetime.now().isoformat(),
        "conditions": conditions,
        "best_config": best_config,
        "comparison": comparison,
    }
    _write_json(destination / "approved_manifest.json", approved_manifest)
    return {**approved_manifest, "approved_dir": str(destination)}
