def assess_phone(phone, evidence, calibrator=None):
    result = {"AccuracyScore": None, "confidence": "low", "status": "uncertain"}
    if calibrator is None:
        return result
    p = calibrator.probability(evidence)
    result["AccuracyScore"] = round(p * 100, 2)
    policy = calibrator.policy
    support = policy.get("phone_support", {}).get(phone, {})
    reliable = (evidence.alignment_sane and policy.get("dev_ece", 1) <= 0.1
                and support.get("correct", 0) >= 20 and support.get("incorrect", 0) >= 10)
    if not reliable:
        return result
    if p >= policy["good_threshold"]:
        result.update(confidence="high", status="ok")
    elif p <= policy["bad_threshold"]:
        result.update(confidence="high", status="needs_practice")
        if evidence.deletion_lpr >= policy["deletion_threshold"]:
            result["status"] = "possibly_missing"
        elif evidence.alternative and evidence.alternative_margin >= policy["substitution_margin"]:
            result["possible_substitute"] = evidence.alternative
    return result
