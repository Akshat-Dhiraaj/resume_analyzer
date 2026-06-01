"""Budget-safe subset runner — 3 AI candidates vs the AI JD.
Showcases the new evidence fields (demonstrated / claimed_only / skill_evidence)
and cert enrichment. Writes to subset_results.json (does not touch the main file)."""
import json
import time
from pathlib import Path

from resume_analyser import ResumeAnalyser, JDParser
from test_analyser import read_pdf, get_or_parse_requirements, run_one, JD_DIR, PROFILES_DIR

JD_FILE = "JD-AI & Computer Vision Engineer (2 Years Experience) – Medical AI 1.pdf"
ROLE    = "AI & Computer Vision Engineer"
FOLDER  = "AI"

def main():
    analyser  = ResumeAnalyser()
    jd_parser = JDParser()
    requirements = get_or_parse_requirements(jd_parser, JD_FILE, read_pdf(JD_DIR / JD_FILE))
    print(f"   required ({len(requirements.required_skills)}): {requirements.required_skills}")

    rows = []
    for resume_path in sorted((PROFILES_DIR / FOLDER).glob("*.pdf")):
        try:
            rows.append(run_one(analyser, resume_path, requirements, ROLE))
        except Exception as e:
            print(f"  !! ERROR: {type(e).__name__}: {e}")
            rows.append({"resume": resume_path.name, "error": f"{type(e).__name__}: {e}"})
        Path("subset_results.json").write_text(json.dumps(rows, indent=2, default=str))
        time.sleep(20)
    print(f"\nSaved {len(rows)} entries to subset_results.json")

if __name__ == "__main__":
    main()
