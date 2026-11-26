"""One-command structure import, preflight, agent and local worker workflow."""
import json
import time

from .gateway import Gateway
from .harness import Harness
from .science import Registry, import_structure
from .worker import run_one


def run_structure(store, model, path, request, session, *, registry=None,
                  potential=None, allowed_operations=None, max_steps=20, max_jobs=8, wall_seconds=600,
                  job_timeout=600, progress=True):
    sid = import_structure(store, path)
    gateway = Gateway(store, Registry(registry))
    checks = {"inspection": gateway.call("inspect_structure", {"structure_id": sid}, session),
              "potentials": gateway.call("list_potentials", {"query": potential or ""}, session)}
    if potential:
        checks["compatibility"] = gateway.call(
            "check_potential", {"structure_id": sid, "potential_id": potential}, session)
    for name, result in checks.items():
        if not result.get("ok"):
            return {"status": "preflight_failed", "structure_id": sid,
                    "preflight": checks, "error": {"stage": name, **result.get("error", {})}}

    context = (f"Imported input structure ID: {sid}\n"
               "Use this registered ID for the user's input structure.\n"
               + (f"Use potential_id {potential}.\n" if potential else
                  "Choose a compatible registered potential; do not invent one.\n")
               + "Preflight observations: " + json.dumps(checks) + "\n"
               + "Retrieve calculation results with get_result before reporting.\n"
               + "User request:\n" + request)
    constraints = {}
    if potential is not None:
        constraints["potential_id"] = potential
    if allowed_operations is not None:
        constraints["allowed_operations"] = allowed_operations
    harness = Harness(store, gateway, model, max_steps, max_jobs, constraints=constraints)
    deadline = time.monotonic() + wall_seconds
    state = harness.run(session, context)
    while state["status"] in ("thinking", "waiting"):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if state["status"] == "waiting":
            # CLI gives each invocation an isolated workspace, so no unrelated
            # queued jobs can be consumed by this local worker.
            worked = run_one(store, min(job_timeout, remaining), show_progress=progress)
            if not worked:
                time.sleep(min(0.1, remaining))
        if time.monotonic() >= deadline:
            break
        state = harness.run(session)

    results = [gateway.call("get_result", {"job_id": job}, session) for job in state["jobs"]]
    status = state["status"]
    if status in ("thinking", "waiting"):
        status = "wall_time_exhausted"
    elif status == "completed" and not results:
        status = "no_calculation"
    elif status == "completed" and any(r.get("status") != "completed" for r in results):
        status = "calculation_failed"
    return {"status": status, "structure_id": sid, "workspace": str(store.root),
            "preflight": checks, "session": state, "results": results,
            "report": store.get_json(state["report_id"], "report") if state["report_id"] else None}
