import datetime

_sample_audits = [
    {
        "id": 1,
        "site": "Site A",
        "company": "ACME Constructions",
        "date": str(datetime.date.today()),
        "total_findings": 4,
        "summary": "Loose scaffolding, missing PPE, exposed wiring",
        "thumbnail": None
    },
    {
        "id": 2,
        "site": "Site B",
        "company": "BuildIt Ltd.",
        "date": str(datetime.date.today()),
        "total_findings": 2,
        "summary": "Trip hazards, poor signage",
        "thumbnail": None
    }
]


def list_audits():
    # Return lightweight list for UI
    return [{
        "id": a["id"],
        "site": a["site"],
        "company": a["company"],
        "date": a["date"],
        "total_findings": a["total_findings"],
        "summary": a["summary"]
    } for a in _sample_audits]


def get_audit(audit_id: int):
    for a in _sample_audits:
        if a["id"] == audit_id:
            # return richer detail
            return {
                "id": a["id"],
                "site": a["site"],
                "company": a["company"],
                "date": a["date"],
                "total_findings": a["total_findings"],
                "summary": a["summary"],
                "findings": [
                    {"sno": 1, "severity": "High", "heading": "Loose scaffolding", "one_liner": "Scaffold connections not secured."},
                    {"sno": 2, "severity": "Medium", "heading": "Missing PPE", "one_liner": "Workers observed without helmets."}
                ]
            }
    return None
