#!/usr/bin/env python3
"""Evaluate captured PR attempts offline. See test_m3_pr_timing.py for snapshot examples.

A snapshot contains as_of, a dated policy (required job names by workflow), a
request, and attempts. Each attempt carries the request's four correlation keys
plus its workflow, run_id, attempt, timestamps, status and jobs. Preserve the
job's actual checkout_sha separately from the PR head. Use null for missing
observations. published_at is the observed required check publication time.
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta
import json
from pathlib import Path

LIMIT = timedelta(minutes=25)
KEYS = ("repository", "pr", "head_sha", "event_id")
WORKFLOWS = {"ci", "approval-storage-evidence"}


def instant(value):
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        raise ValueError("timestamps must include a timezone")
    return parsed


def evaluate(snapshot):
    request = snapshot["request"]
    attempts = snapshot["attempts"]
    now = instant(snapshot["as_of"])
    start = instant(request.get("requested_at"))
    superseded = instant(request.get("superseded_at"))
    if now is None or any(t > now for t in (start, superseded) if t is not None):
        raise ValueError("request timestamps exceed the observation window")
    if start and superseded and superseded < start:
        raise ValueError("supersession precedes the request")

    # Keep observations, including rerun attempts, available for inspection.
    result = {"request": request, "attempts": attempts, "policy": snapshot["policy"],
              "as_of": snapshot["as_of"], "deadline": "unscorable",
              "verdict": "unknown", "required_checks": [], "superseded": bool(superseded)}
    if request.get("event") != "pull_request" or request.get("same_repository") is not True:
        result["deadline"] = "ineligible"
        return result
    policy = snapshot["policy"]
    required = policy["required"]
    captured = instant(policy.get("captured_at"))
    if (captured is None or captured > now or set(required) != WORKFLOWS
            or any(not isinstance(names, list) or not names or len(set(names)) != len(names)
                   for names in required.values())
            or "approval-storage-macos-arm64" not in required["ci"]
            or "approval-storage-macos-standalone" in required["approval-storage-evidence"]):
        raise ValueError("expected a dated same-repository PR check policy")
    if not all(request.get(k) for k in KEYS):
        result["reason"] = "missing request identity"
        return result

    records = {}
    checks = {workflow: {} for workflow in required}
    queued, updated = [], []
    for attempt in sorted(attempts, key=lambda a: a["attempt"]):
        workflow = attempt["workflow"]
        if workflow not in required or any(attempt.get(k) != request[k] for k in KEYS):
            result["reason"] = "ambiguous request or workflow identity"
            return result
        previous = records.get(workflow)
        if previous and (previous["run_id"] != attempt["run_id"]
                         or previous["attempt"] == attempt["attempt"]):
            result["reason"] = "conflicting workflow attempts"
            return result
        records[workflow] = attempt
        for field, dest in (("created_at", queued), ("updated_at", updated)):
            t = instant(attempt.get(field))
            if t is not None:
                if t > now:
                    raise ValueError("workflow timestamp exceeds the observation window")
                dest.append(t)

        if attempt["attempt"] < 1 or not attempt["run_id"]:
            raise ValueError("invalid attempt identity")
        # A failed-jobs-only rerun may omit successful checks. Carry a success
        # only across adjacent attempts of this run, after the retry settles.
        inherited = checks[workflow]
        if (not previous or previous["attempt"] + 1 != attempt["attempt"]
                or attempt["status"] != "completed"):
            inherited = {}
        jobs = {n: j for n, j in inherited.items() if j["conclusion"] == "success"}
        seen = set()
        for job in attempt["jobs"]:
            name = job["name"]
            if name in seen:
                result["reason"] = "duplicate job name in workflow attempt"
                return result
            seen.add(name)
            if name not in required[workflow]:
                continue
            timeline = [instant(job.get(k)) for k in
                        ("created_at", "started_at", "completed_at", "published_at")]
            times = [t for t in timeline if t is not None]
            if times != sorted(times) or any(t > now for t in times):
                raise ValueError("invalid job timeline")
            jobs[name] = {**job, "workflow": workflow, "run_id": attempt["run_id"],
                          "attempt": attempt["attempt"]}
        checks[workflow] = jobs

    if (len(queued) == len(updated) == len(attempts) and set(records) == WORKFLOWS
            and all(a["status"] == "completed" for a in records.values())):
        result["workflow_creation_to_update_proxy_seconds"] = (max(updated) - min(queued)).total_seconds()

    rows = [job for jobs in checks.values() for job in jobs.values()]
    missing = [[workflow, name] for workflow, names in required.items()
               for name in names if name not in checks[workflow]]
    result.update(required_checks=rows, missing_checks=missing)
    publications = [instant(row.get("published_at")) for row in rows]
    if any(t > now or (start and t < start) for t in publications if t is not None):
        raise ValueError("check publication falls outside request window")
    result["missing_publication_count"] = publications.count(None)
    result["missing_request_timestamp"] = start is None
    # A superseded request is evaluated at supersession; late rerun results
    # cannot turn a still-pending request back into an eligible approval.
    terminal = (not missing and rows and all(row["status"] == "completed" for row in rows))
    if terminal and (not superseded or all(t and t <= superseded for t in publications)):
        outcomes = {row["conclusion"] for row in rows}
        if outcomes <= {"success", "failure"}:
            result["verdict"] = "rejected" if "failure" in outcomes else "approved"
        else:
            result["verdict"] = "inconclusive"

    if start is None:
        result["reason"] = "missing request timestamp"
    elif result["verdict"] in ("approved", "rejected"):
        if all(publications):
            elapsed = max(publications) - start
            result.update(elapsed_seconds=elapsed.total_seconds(),
                          deadline="on_time" if elapsed <= LIMIT else "miss")
        else:
            result["reason"] = "missing required publication timestamp"
    elif result["verdict"] == "inconclusive":
        result["deadline"] = "miss"
    elif superseded:
        result["deadline"] = "superseded_before_deadline" if superseded - start < LIMIT else "miss"
    elif now - start >= LIMIT:
        result["deadline"] = "miss"
    else:
        result["deadline"] = "pending"
    return result


def cohort(snapshots):
    results = []
    seen = set()
    for snapshot in snapshots:
        result = evaluate(snapshot)
        identity = tuple(result["request"].get(k) for k in KEYS)
        if all(identity):
            if identity in seen:
                raise ValueError("duplicate logical request; combine its attempts before scoring")
            seen.add(identity)
        results.append(result)
    counts = Counter(r["deadline"] for r in results)
    return {"requests": results, "counts": dict(counts),
            "superseded_requests": sum(r["superseded"] for r in results),
            "scored_deadline_denominator": counts["on_time"] + counts["miss"],
            "missing_request_timestamps": sum(not r["request"].get("requested_at") for r in results),
            "missing_required_publications": sum(r.get("missing_publication_count", 0) for r in results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, nargs="+", help="one JSON snapshot per logical request")
    args = parser.parse_args()
    try:
        result = cohort(json.loads(path.read_bytes()) for path in args.snapshot)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"invalid timing snapshot: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
