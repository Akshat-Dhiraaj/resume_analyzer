# Resume Analyser

A DSPy + Groq pipeline that scores a resume against a job description and produces a fit assessment with concrete improvement suggestions. Built around the principle that **the LLM extracts atoms; Python does the arithmetic, the matching, the seniority alignment, and the hallucination filtering**.

## What it does

Given a resume PDF and a job description PDF, the pipeline produces:

- A structured **candidate profile** — name, email, dated work history, technical skills, education — with multiple Python passes guaranteeing every emitted skill is grounded in the source text.
- A **years-of-experience** number computed from the actual work-history dates (merge-intervals over parsed dates, not the LLM's guess).
- A **years-of-relevant-experience** number, weighted by how much of the candidate's stack overlaps with the JD's required skills via a curated synonym map.
- A **fit assessment** with three sub-scores (skills / experience / seniority, each 0-100), all computed in Python and combined deterministically into an overall 0-100 score. The LLM's only role at this stage is to write a 2-3 sentence rationale narrating the numbers.
- **3-5 score-calibrated suggestions** — pivot recommendations for poor fits, gap-filling for moderate fits, polish for strong fits.

## Pipeline

```
   PDF (resume)                 PDF (job description)
        │                                │
        ▼                                ▼
     pymupdf                          pymupdf
        │                                │
        │                          ┌───────────────┐
        │                          │ JDParser      │ ◄── cached on disk
        │                          │ (CoT)         │     (requirements_cache.json)
        │                          └───────────────┘
        │                                │
        │                          JobRequirements {
        │                            role_title, seniority,
        │                            min_years_experience,
        │                            required_skills,
        │                            nice_to_have_skills,
        │                            responsibilities,
        │                          }
        ▼                                │
   ┌────────────────────────┐            │
   │ ExtractProfile (CoT)   │            │
   └────────────────────────┘            │
        │                                │
        ▼                                │
   ┌──────────────────────────────────┐  │
   │ Python deterministic passes:     │  │
   │  • clean_name (kerning repair)   │  │
   │  • drop_empty_roles              │  │
   │  • filter_education              │  │
   │  • recover_missed_skills         │  │
   │  • recover_email                 │  │
   │  • filter_hallucinated_extraction│  │
   │  • extract_claimed_years         │  │
   │  • compute_years_experience      │  │
   │  • reconcile_years               │  │
   │  • compute_relevant_years        │  │
   │  • match_skills (required+nice)  │  │
   │  • compute_skills_match_score    │  │
   │  • compute_experience_match_score│  │
   │  • compute_seniority_score       │  │
   └──────────────────────────────────┘  │
        │                                │
        ▼                                ▼
   ┌────────────────────────────────────────┐
   │ AssessFit (CoT)                        │
   │   ↳ rationale (narrates the scores)    │
   │   (all numbers + matched/missing fed   │
   │    in as Python-computed inputs)       │
   └────────────────────────────────────────┘
        │
        ▼
   FitAssessment  ◄── combine_subscores → overall_score
        │
        ▼
   ┌────────────────────────────────────────┐
   │ GenerateAdvice (CoT, creative_lm)      │
   │   branches on overall_score:           │
   │    < 50   → pivot recommendation       │
   │    50-69  → fill specific gaps         │
   │    ≥ 70   → polish + quantify          │
   └────────────────────────────────────────┘
        │
        ▼
   list[str] suggestions
```

## Files in this project

| File | Purpose |
|---|---|
| `.env` | Holds `GROQ_API_KEY=gsk_...` (loaded by `config.py` via `python-dotenv`) |
| `config.py` | Reads `GROQ_API_KEY` from the environment |
| `resume_analyser.py` | Schemas, signatures, all Python helpers, the two DSPy modules |
| `test_analyser.py` | Runner: reads PDFs, caches JD parses, runs the pipeline, handles 429s, saves JSON |
| `requirements_cache.json` | Auto-created — parsed `JobRequirements` per JD, reused across runs |
| `analyser_results.json` | Auto-created — per-candidate profile + assessment + suggestions, written incrementally |
| `ARCHITECTURE.md` | What's handled by what, and why |

## How to run

```powershell
pip install -r requirements.txt
# Put your key in .env:  GROQ_API_KEY=gsk_...
python test_analyser.py
```

The fixture has 3 JDs (Full Stack / AI-CV / QA) and 10 candidate resumes split by target role.

**Important: do not `pip install fitz`.** That's an unrelated PyPI package. The PyMuPDF library installs as the package `PyMuPDF` but you `import fitz` to use it.

## Models

Two Groq models, chosen per stage based on what the LLM is being asked to do.

| LM | Model | Temperature | Used for |
|---|---|---|---|
| `precise_lm` | `llama-3.3-70b-versatile` | 0.0 | JD parse, profile extraction, rationale narration — stages where accuracy and reproducibility matter |
| `creative_lm` | `llama-3.1-8b-instant` | 0.2 | `GenerateAdvice` only — generative writing that doesn't justify 70B tokens, and where mild variability is desirable |

The hybrid split keeps a 10-candidate run inside Groq's free-tier **100K TPD** cap on the 70B model (extraction is the heaviest stage; advice on 8B saves roughly 30% of the daily budget). The Python defence-in-depth passes (verbatim filter, recovery, strict-subset) stay in place regardless of model size — they're additive, not corrective, and they keep the pipeline reproducible across re-runs.

## Key design decisions

### Hybrid LLM/Python by stage

The original pipeline let the LLM produce everything; the current version moves anything with a deterministic answer to Python:

| Concern | Owner |
|---|---|
| JD parse | LLM |
| Profile extraction (structure) | LLM |
| Required-skill matching | **Python** (set arithmetic + curated `SYNONYMS`) |
| Years of experience | **Python** (merge-intervals over parsed dates) |
| Years of *relevant* experience | **Python** (total × match ratio) |
| Seniority alignment | **Python** (title rank vs JD-target rank) |
| Claimed-years extraction | **Python** (regex over resume header) |
| Hallucination filter | **Python** (verbatim + strict-subset against `resume_text`) |
| 8B-drop recovery | **Python** (SYNONYMS lexicon scan) |
| skills_match_score | **Python** (`100 · matched / required`) |
| experience_match_score | **Python** (ratio vs min, capped at 50 when `dates_extracted_ok=False`) |
| seniority_match_score | **Python** |
| overall_score | **Python** (`0.5·skills + 0.3·exp + 0.2·seniority`) |
| Rationale | LLM (narrates the Python-computed scores) |
| Suggestions (score-calibrated) | LLM |

### Hallucination defences

8B is cheaper but more prone to invention and field-dropping than 70B. Three Python passes neutralise this:

- **`recover_missed_skills`** — scans `resume_text` against every alias in `SYNONYMS` with word-boundary matching; any canonical skill whose alias appears in the source but isn't in `technical_skills` gets added back. Catches Divya's collapsed `["Testing", "Tools", "Database"]` and Shivkumar's lost `Express.js`.
- **`filter_hallucinated_extraction`** — three passes: (1) drops every emitted skill that doesn't appear *anywhere* in `resume_text` (verbatim filter); (2) enforces `work_history[].stack ⊆ technical_skills` (strict subset), catching Sarthak's IDEMIA stack being grafted with `React`/`Node.js`/`Docker` despite him being a Java/Spring/Angular dev; (3) `_dedupe` collapses exact duplicates and canonical-vs-detail variants (`AWS` + `AWS (EC2, S3, IAM)`) while keeping distinct products (`Spring` + `Spring Boot`).
- **`recover_email`** — regex fallback when the LLM loses the address to a kerning artefact like Shivkumar's `R jztshiva@gmail.com`.

### All sub-scores in Python; LLM only writes the rationale

Every numeric score — skills, experience, seniority, overall — is computed deterministically in Python from the matched/missing lists and the year metrics. Across multiple runs the LLM consistently inflated `skills_match_score` (Aakash 5/8 matched → 85; Divya 8/14 → 85) even when handed Python-computed matched/missing as input. Moving the formula to Python ends that. `AssessFit` is now a single-output module: a 2-3 sentence rationale that references specific skills and acknowledges unverified dates when `dates_extracted_ok` is False.

### Date-extraction integrity flag

`dates_present()` returns False when (a) zero roles have a usable start_date, or (b) >1 role have non-empty starts that are all identical, or (c) >1 role have non-empty starts and all end in `present`. Partial date data is OK (e.g. one verified role + one open-source contribution with empty dates). When the flag is False, `compute_experience_match_score` caps the result at 50.

### One LM with two temperatures

`precise_lm` is the default for everything that needs reproducibility (JD parse, extract, assess). `creative_lm` is reserved for advice generation. No 70B → no daily token-cap blocker.

### Synonym-aware matching

`match_skills` returns `(matched, missing)` lists derived from set membership against an expanded alias table (e.g., `Selenium WebDriver` → matches `Selenium`; `AWS (EC2, S3, …)` → matches `AWS`; `Tailwind CSS` → matches `CSS3`). The matched list uses the JD's canonical wording, so downstream reports are consistent across candidates for the same JD.

## Results

10 Naukri-format resumes against 3 JDs (Full Stack / AI-CV / QA). The deterministic passes produce stable, differentiated scores; re-runs of the same candidate give the same answer because the only LLM stage that touches resume content is extraction, which is now backed by the verbatim filter and recovery passes.

Track per-candidate output in `analyser_results.json`. To produce a ranked shortlist per JD, sort by `assessment.overall_score`.

## Known limitations

- **Synonym map is hand-rolled.** Covers the fixture comfortably but won't generalise to every JD. Future work: replace `SYNONYMS` with the EMSI/Lightcast skill DB bundled by [SkillNER](https://github.com/AnasAito/SkillNER) (~34k skills with synonyms).
- **No DSPy `Refine` retry loop yet.** The verbatim filter is a silent drop; wrapping `ExtractProfile` in `dspy.Refine` with a reward function that asserts "every claimed skill appears in resume_text" would convert it to a feedback loop. See [DSPy assertions docs](https://dspy.ai/learn/programming/7-assertions/).
- **No automated optimisation.** `BootstrapFewShot` / `MIPROv2` aren't wired in. Add once you have ~15+ hand-labelled (resume, JD, expected sub-scores) triples.
- **Output is per-candidate, not ranked.** Each candidate is scored independently. For a ranked leaderboard, sort `analyser_results.json` by `assessment.overall_score` per JD — that's a one-liner.

## What this is deliberately *not*

- Not a prompt-tuning framework — instructions live in Signature docstrings (single source of truth).
- Not a CV generator — it analyses, it doesn't rewrite.
- Not a sourcing tool — assumes you already have the candidates.
