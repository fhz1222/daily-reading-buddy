def aggregate(assessments):
    scores = [a.get("AccuracyScore") for a in assessments]
    status = "uncertain"
    statuses = {a["status"] for a in assessments}
    if statuses == {"possibly_missing"}:
        status = "possibly_missing"
    elif statuses & {"needs_practice", "possibly_missing"}:
        status = "needs_practice"
    elif statuses == {"ok"}:
        status = "ok"
    return {"AccuracyScore": round(sum(scores) / len(scores), 2) if scores and all(s is not None for s in scores) else None,
            "status": status}
