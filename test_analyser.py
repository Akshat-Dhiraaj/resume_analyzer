"""test_analyser.py — runs ResumeAnalyser over the Profiles/ fixture set.

Each JD is parsed into JobRequirements once (cached on disk) and reused
across every candidate folder, so the LLM only sees a given JD once."""

import json
import re
import sys
import time
from pathlib import Path
import fitz  # pymupdf

# Force UTF-8 on Windows so the formatted output (em-dashes, ⏸, ⚠️, etc.)
# doesn't crash on cp1252 consoles.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from resume_analyser import ResumeAnalyser, JDParser, JobRequirements

ROOT          = Path("Profiles")
JD_DIR        = ROOT / "JD"
PROFILES_DIR  = ROOT / "Profiles"
CACHE_PATH    = Path("requirements_cache.json")
RESULTS_PATH  = Path("analyser_results.json")
JD_COOLDOWN_S = 15        # pause after parsing a JD before hitting candidates
CANDIDATE_PACING_S = 20   # pause between candidates to stay under TPM limits

JD_TO_FOLDER = {
    "JD- Full Stack Developer.pdf": ("SDE", "Full Stack / MERN Developer"),
    "JD-AI & Computer Vision Engineer (2 Years Experience) – Medical AI 1.pdf":
        ("AI", "AI & Computer Vision Engineer"),
    "Job Description -QA Engineer.pdf": ("QA", "QA Engineer"),
}

_RATE_LIMIT_HINT = re.compile(r"try again in ([\d.]+)s")

def call_with_retry(fn, *args, max_retries: int = 5, **kwargs):
    """Retry on Groq 429s, parsing the API's suggested wait time. Re-raises
    the final exception after exhausting retries (no silent extra call)."""
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if "rate_limit" not in str(e).lower():
                raise
            m = _RATE_LIMIT_HINT.search(str(e))
            wait = float(m.group(1)) + 3 if m else 30.0
            print(f"  ⏸  rate-limited, sleeping {wait:.1f}s (attempt {attempt}/{max_retries})")
            time.sleep(wait)
    raise last_exc  # type: ignore[misc]

def read_pdf(path: Path) -> str:
    return "\n".join(p.get_text() for p in fitz.open(str(path)))

def save_results(rows: list[dict]) -> None:
    RESULTS_PATH.write_text(json.dumps(rows, indent=2, default=str))

def get_or_parse_requirements(jd_parser: JDParser, jd_file: str, jd_text: str):
    """Return cached JobRequirements if present, else LLM-parse and cache."""
    cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    if jd_file in cache:
        return JobRequirements(**cache[jd_file])
    print(f"   (parsing JD — not in cache)")
    req = call_with_retry(jd_parser, jd_text=jd_text)
    cache[jd_file] = req.model_dump()
    CACHE_PATH.write_text(json.dumps(cache, indent=2, default=str))
    return req

def run_one(analyser, resume_path: Path, requirements, target_role: str):
    print(f"\n{'='*72}")
    print(f"▶ {resume_path.name}  vs  {target_role}")
    print('='*72)
    resume_text = read_pdf(resume_path)
    print(f"  resume chars: {len(resume_text):,}")

    t0 = time.time()
    result = call_with_retry(
        analyser,
        resume_text=resume_text,
        requirements=requirements,
        target_role=target_role,
    )
    dt = time.time() - t0

    p, a = result.profile, result.assessment
    dates_flag = "" if p.dates_extracted_ok else "  ⚠️ NO DATES EXTRACTED"
    print(f"  ⏱ {dt:.1f}s")
    print(f"\n  ── Profile ──")
    print(f"  name             : {p.name}")
    print(f"  email            : {p.email}")
    print(f"  years_experience : {p.years_experience}{dates_flag}")
    print(f"  years_relevant   : {p.years_relevant_experience}")
    print(f"  technical_skills : {len(p.technical_skills)} skills")

    print(f"\n  ── Fit Assessment ──")
    print(f"  skills_match     : {a.skills_match_score}/100")
    print(f"  experience_match : {a.experience_match_score}/100")
    print(f"  seniority_match  : {a.seniority_match_score}/100")
    print(f"  ─────────────────────")
    print(f"  OVERALL          : {a.overall_score}/100")
    print(f"  matched_required : {a.matched_required}")
    print(f"  missing_required : {a.missing_required}")
    print(f"  rationale        : {a.rationale}")

    print(f"\n  ── Suggestions ──")
    for i, s in enumerate(result.suggestions, 1):
        print(f"  {i}. {s}")

    return {
        "resume": resume_path.name,
        "target_role": target_role,
        "elapsed_sec": round(dt, 1),
        "profile": p.model_dump(),
        "assessment": a.model_dump(),
        "suggestions": result.suggestions,
    }

def main():
    analyser = ResumeAnalyser()
    jd_parser = JDParser()
    all_results: list[dict] = []

    for jd_file, (folder, role) in JD_TO_FOLDER.items():
        jd_path = JD_DIR / jd_file
        if not jd_path.exists():
            print(f"!! missing JD: {jd_path}")
            continue

        print(f"\n📋 JD: {jd_file}")
        requirements = get_or_parse_requirements(jd_parser, jd_file, read_pdf(jd_path))
        nice = requirements.nice_to_have_skills
        print(f"   role: {requirements.role_title}  ({requirements.seniority}, min {requirements.min_years_experience}y)")
        print(f"   required ({len(requirements.required_skills)}): {requirements.required_skills}")
        print(f"   nice-to-have ({len(nice)}): {nice[:5]}{'...' if len(nice) > 5 else ''}")
        time.sleep(JD_COOLDOWN_S)

        for resume_path in sorted((PROFILES_DIR / folder).glob("*.pdf")):
            try:
                all_results.append(run_one(analyser, resume_path, requirements, role))
            except Exception as e:
                print(f"  !! ERROR: {type(e).__name__}: {e}")
                all_results.append({
                    "resume": resume_path.name,
                    "target_role": role,
                    "error": f"{type(e).__name__}: {e}",
                })
            save_results(all_results)
            time.sleep(CANDIDATE_PACING_S)

    # No trailing save: save_results runs after every candidate, so the file
    # is already current at this point.
    print(f"\n\nSaved {len(all_results)} entries to {RESULTS_PATH.resolve()}")

if __name__ == "__main__":
    main()
