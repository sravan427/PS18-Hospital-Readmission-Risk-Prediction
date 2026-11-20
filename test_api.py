import json
import math
import requests

API_URL = "http://127.0.0.1:8000/predict"
CASES_FILE = "api_validation_cases.json"
TOLERANCE = 0.00011  # API and notebook probabilities are rounded to 4 decimals.

with open(CASES_FILE, "r", encoding="utf-8") as f:
    cases = json.load(f)

results = []

for case in cases:
    payload = case["payload"]
    expected = case["expected"]

    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        response.raise_for_status()
        actual = response.json()
        

        api_probability = float(actual["readmission_probability"])
        notebook_probability = float(expected["probability"])

        probability_match = math.isclose(
            api_probability, notebook_probability,
            rel_tol=0.0, abs_tol=TOLERANCE
        )
        risk_match = actual.get("risk_level") == expected["risk_level"]
        flag_match = actual.get("flagged_high_risk") == expected["flagged_high_risk"]

        results.append({
            "encounter_id": expected["encounter_id"],
            "notebook_probability": notebook_probability,
            "api_probability": api_probability,
            "probability_match": probability_match,
            "notebook_risk": expected["risk_level"],
            "api_risk": actual.get("risk_level"),
            "risk_match": risk_match,
            "notebook_flag": expected["flagged_high_risk"],
            "api_flag": actual.get("flagged_high_risk"),
            "flag_match": flag_match,
            "overall": "MATCH" if probability_match and risk_match and flag_match else "MISMATCH"
        })

    except Exception as exc:
        results.append({
            "encounter_id": expected["encounter_id"],
            "overall": f"ERROR: {exc}"
        })

print("\nFASTAPI vs NOTEBOOK VALIDATION")
print("-" * 90)
for row in results:
    print(row)

valid = [r for r in results if r.get("overall") in ("MATCH", "MISMATCH")]
matches = sum(r["overall"] == "MATCH" for r in valid)

print("\nSUMMARY")
print(f"Successful API responses: {len(valid)}/{len(cases)}")
print(f"Exact matches: {matches}/{len(cases)}")
print(f"Mismatches: {len(valid) - matches}/{len(cases)}")

if valid and matches == len(cases):
    print("SUCCESS: All tested FastAPI predictions match the notebook.")
else:
    print("CHECK REQUIRED: Inspect mismatched rows and the preprocessing in main.py.")