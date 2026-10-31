def updated_trust(current, decision, validation_summary):
    # Ambiguous review is a small penalty; confirmed/quarantined event is larger.
    delta = 1.0 if decision == "ACCEPTED" else (-3.0 if decision == "NEEDS REVIEW" else -12.0)
    if validation_summary["rejected"]: delta -= min(4, validation_summary["rejected"] / max(1, validation_summary["total"])*20)
    return max(0., min(100., current + delta)), delta
