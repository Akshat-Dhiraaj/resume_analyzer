# Architecture — what's handled by what

This document maps every meaningful concern in the pipeline to the specific
piece of code that handles it, and explains why. It complements `README.md`,
which gives the high-level flow.

## Responsibility map

| Concern | Handled by | Why this choice |
|---|---|---|
| PDF binary → text | `fitz.open(path).get_text()` in `test_analyser.py::read_pdf` | pymupdf preserves spacing on kerned PDFs; pypdf produces `"tarunupadha y a y95@gmail.c om"` on stylised resumes |
| Parsing the JD into a structured contract | `dspy.ChainOfThought(ParseJobDescription)` wrapped in `JDParser` module | Without a canonical required-skills list, every candidate assessment has to re-invent its own list, producing inconsistent matched/missing keywords across candidates |
| Repairing kerned name (`A B H I S H E K  M A J H I` → `ABHISHEK MAJHI`) | `clean_name()` | PDF kerning artefacts leak into the LLM's extracted name. Single regex collapse, only triggers when ≥60% of tokens are single characters |
| Dropping placeholder work_history entries with blank role/company | `drop_empty_roles()` | 8B occasionally promotes a course/open-source line into work_history with empty fields. Empty-role entries break the seniority inference and pollute the JSON |
| Filtering education to post-secondary entries | `filter_education()` keyed off `_DEGREE_KEYWORDS` | The LLM keeps school entries despite the docstring telling it not to. Cheaper to filter in Python than to retry |
| Dropping job titles mis-filed as certifications | `drop_non_certifications()` — keep credential-worded names, drop role-shaped ones (`intern`/`engineer`/…) that lack a credential word | The LLM filed "Data Science Intern" as a cert, which polluted both the output and `cert_knowledge.json`. Credential words (`certified`/`professional`/`associate`/…) override the role check, so "AWS Certified Developer - Associate" survives |
| Caching parsed JDs across runs | `get_or_parse_requirements()` in `test_analyser.py`, writes to `requirements_cache.json` | JD content doesn't change between candidate evaluations; saves 1 LLM call per JD per run |
| Distinguishing required vs nice-to-have skills | JDParser puts mandatory items in `required_skills`, optional/preferred items in `nice_to_have_skills` | Equal weighting of "must have AWS" and "Kafka is a plus" would distort scoring |
| Identifying skills, education, work entries (structure only) | `dspy.ChainOfThought(ExtractProfile)` | The LLM produces the shape; deterministic passes correct the content |
| Recovering LLM-dropped skills | `recover_missed_skills()` — scans `resume_text` against the `SYNONYMS` alias map with word-boundary matching | 8B occasionally collapses `["Functional Testing", "Regression Testing", ...]` to `["Testing", "Tools", "Database"]` (Divya's case). Lexicon recovery rebuilds the flat inventory from the source |
| Recovering LLM-dropped email | `recover_email()` — regex fallback on `resume_text` | 8B has dropped email on glyph-corrupted PDFs ("R jztshiva@..." → no email). Single-purpose regex fallback |
| Dropping fabricated skills | `filter_hallucinated_extraction()` pass 1 — verbatim check against normalised `resume_text` | Anything the LLM emits that isn't grounded in the source is silently removed |
| Dropping mis-attributed work-history stack | `filter_hallucinated_extraction()` pass 2 — `stack ⊆ technical_skills` (strict subset) | The LLM tends to graft `React`/`Node.js`/`Docker` onto every role's stack. Sarthak (Java/Spring/Angular at IDEMIA) was being scored as a MERN dev. Strict subset forces the role's stack to be a subset of what the candidate even claims as a skill |
| Collapsing skill duplicates and canonical-vs-detail variants | `filter_hallucinated_extraction()` pass 3 — `_dedupe()` with `_bare()` for paren-stripped comparison | The 70B extractor sometimes lists the same skill twice in a role's stack (Sarthak: `Spring Boot ×2`); recovery adds canonical names that overlap with detail-bearing variants the LLM already emitted (Vishal: `AWS` + `AWS (EC2, S3, IAM)`; Yash: `Python` + `Python (Basic)`). Order-preserving — keeps the first/richer occurrence. Distinct products (`Spring` + `Spring Boot`) are NOT collapsed |
| Parsing date strings (`"July 2023"`, `"2023-02"`, `"present"`) | `_parse_ym()` using `dateutil.parser.parse(fuzzy=True)` | One source of truth, handles natural-language variants |
| Summing across multiple roles (total years) | `compute_years_experience()` — sort + merge-intervals | LLMs are unreliable at multi-step date arithmetic; merging overlap is needed because candidates list concurrent roles (full-time + freelance), and naïve summing double-counts |
| Detecting genuine vs. hallucinated dates | `dates_present()` — no usable starts → False; >1 role ALL with non-empty starts AND all identical → False; >1 role ALL with non-empty starts AND all ending in `present` → False | Abhilash's case: 4 roles all returned start='2018' / end='present'. Partial data (one verified role + one role with empty dates) is still considered valid |
| Reconciling computed years vs the resume's prose claim | `reconcile_years()` heuristic | LLM-extracted start dates can be wrong; the prose claim is a sanity-check ceiling. Rule: if computed > claimed + 2.0 years, the claim wins |
| Extracting the claimed-years figure | `extract_claimed_years()` — regex over `resume_text[:600]` | Three patterns: Naukri header tag (`3y 1m`), `X+ years experience`, bare `X years`. Reliable enough that the LLM doesn't need to populate `summary_claimed_years` at all |
| Weighting years by relevance to the target JD | `compute_relevant_years()` — total years × (evidence_sum / required_count) | Weighted by the same evidence sum used for the skills score, so a merely-listed skill contributes less relevant time than a demonstrated one. Nice-to-haves are NOT in this calculation: they signal fit-bonus, not relevance |
| Resolving a skill's equivalences | `resolve_skill()` → `ResolveSkillEquivalences` on 70B, cached in `skill_knowledge.json` | Equivalence ("AWS ≈ EC2/S3", "K8s = Kubernetes") is a stable world-fact, not a per-candidate call — resolve once with a conservative prompt and cache. The LLM describes the *skill*, never judges the *candidate* |
| Skill matching against the JD | `match_skills()` / `skill_match_set()` — typed, directional set membership over `skill_knowledge.json` | Earlier LLM-decided matching hallucinated (Yash's `Selenium WebDriver` flagged as missing `Selenium`; Sarthak got false React). Now deterministic: a required skill matches on its `aliases` or `entailed_by` tokens; `related_only` (Docker vs Kubernetes, Java vs JavaScript) never matches. Directional — EC2⇒AWS, not vice-versa |
| Splitting compound skills (`AWS (EC2, S3, IAM)`) into atoms | `_expand_candidate_stack()` — `re.split` on `()&/-,` | Makes synonym matching forgiving of resume formatting without resorting to substring search |
| Scoring each required skill by evidence | `build_evidence_map()` — 0–1 per skill from where it appears (project/role/cert/listed) + corroboration bonus | Binary "has it" rewards a candidate who merely lists AWS the same as one who shipped it. Evidence weighting separates demonstrated depth from self-claims. A skill is evidenced if it's in the skills list, a project, a role, OR a credential that *validates* it (certs expand into their skills, not just their title) |
| Cert tier, reputation, skills, shelf-life | `classify_cert()` + `cert_weight()` over `cert_knowledge.json` (learned instance → `_patterns` official rules → one-time LLM enrichment → keyword heuristic) | Weight depends on REPUTATION not just tier: a professional cert or high-reputation course → 0.95 (beats a project), medium → 0.60, low → 0.45. Unseen certs get a single cheap-model enrichment call (`_enrich_cert_via_llm`) returning reputation + validated skills + validity, cached forever. Dated certs decay past validity (AWS/Red Hat 3y, GCP/CKA 2y, Azure 1y); undated treated as current |
| Computing the skills sub-score | `compute_skills_match_score()` — `100 · mean(evidence)` | The LLM consistently inflated even when given Python-provided matched/missing (Aakash 5/8 → 85, Divya 8/14 → 85). Now an evidence-weighted mean, fully in Python |
| Computing the experience sub-score | `compute_experience_match_score()` — ratio vs min, or absolute-years ladder when min is 0, capped at 50 when `dates_extracted_ok` is False | Same reason: ratio-based. The date-verification cap means Abhilash-class fabricated dates can never produce a high experience score |
| Computing the seniority sub-score | `compute_seniority_score()` — title-rank lookup vs JD-target rank, with at-target / over-/under-qualified branches and a years-meets-min check | The LLM under-scored over-qualified candidates (Vishal: 70, expected 90) and over-scored under-experienced ones at-target |
| Writing the fit rationale | `dspy.ChainOfThought(AssessFit)` — outputs `rationale` only | All sub-scores are Python-computed; the LLM narrates them in 2-3 sentences. Must acknowledge `dates_extracted_ok=False` when set |
| Combining sub-scores into `overall_score` | `combine_subscores()` — Python weighted sum (50/30/20) | Reproducible, debuggable, doesn't depend on LLM whim |
| Generating concrete improvement suggestions | `dspy.ChainOfThought(GenerateAdvice)` on `creative_lm` (temp 0.2) | Generative writing benefits from mild variability; score-band-aware behaviour enforced by the docstring |
| Score-aware advice branching | Pass `overall_score` as input to `GenerateAdvice`; signature docstring describes the three bands | Without this, low-fit candidates got "add Express.js section"; now they get "consider pivoting to a better-aligned role family" |
| Validating LLM output structure | Pydantic `BaseModel` subclasses (`ResumeProfile`, `WorkExperience`, `Project`, `Certification`, `JobRequirements`, `CertInfo`, `_AssessRationale`, `FitAssessment`) | DSPy parses typed outputs natively; on parse failure it retries automatically |
| Score range enforcement (0-100) | `Field(ge=0, le=100)` on every score | Pydantic rejects out-of-range values |
| Stage-level model selection | Default is `precise_lm` (temp 0.0); `dspy.context(lm=creative_lm)` overrides for advice only | Only one override block in `forward()` |
| Groq rate-limit handling | `call_with_retry()` in `test_analyser.py` | Parses Groq's "try again in Xs" hint from 429s; bounded retries; re-raises the last exception explicitly |
| Progress preservation under failure | `save_results()` after every candidate | Long batch runs can hit rate limits; if the process dies we keep what's done |

## Data flow

```
   resume_path : Path                      jd_text : str
        │                                       │
        ▼                                       ▼
   fitz.open() ──► resume_text          JDParser (cached)
                                                │
                                                ▼
                                       JobRequirements {
                                          role_title, seniority,
                                          min_years_experience,
                                          required_skills,
                                          nice_to_have_skills,
                                          responsibilities,
                                       }
        │                                       │
        └────────────────────┬──────────────────┘
                             ▼
   ResumeAnalyser.forward(resume_text, requirements, target_role)
        │
        ├── Phase 1: extract + clean
        │     self.extract(resume_text)
        │     └─► ResumeProfile (raw — may have invented or dropped items)
        │     profile.name         = clean_name(profile.name)
        │     profile.work_history = drop_empty_roles(profile.work_history)
        │     profile.education    = filter_education(profile.education)
        │     recover_missed_skills(profile, resume_text)
        │     recover_email(profile, resume_text)
        │     filter_hallucinated_extraction(profile, resume_text)
        │     profile.summary_claimed_years = extract_claimed_years(resume_text)
        │
        ├── Phase 2: deterministic profile metrics
        │     dates_extracted_ok = dates_present(work_history)
        │     years_experience   = reconcile_years(compute_years_experience(...),
        │                                          summary_claimed_years)
        │
        ├── Phase 3: evidence-weighted matching, relevance, and sub-scores
        │     skill_cache = load_skill_cache()                  # LLM-resolved equivalences
        │     evidence = build_evidence_map(profile, requirements, today, skill_cache)
        │       ├─ skill_knowledge.json for aliases / entailed_by / related_only
        │       └─ cert_knowledge.json  for cert tier + reputation + recency
        │     demonstrated / claimed_only / missing  ← split by evidence threshold
        │     evidence_sum = Σ evidence
        │     matched_nice, _ = match_skills(candidate_stack(profile), nice_to_have_skills, skill_cache)
        │     save_skill_cache(skill_cache)  # if new skills were resolved
        │     years_relevant_experience = compute_relevant_years(years_experience,
        │                                          evidence_sum, len(required_skills))
        │     skills_score     = compute_skills_match_score(evidence_sum, len(required_skills))
        │     experience_score = compute_experience_match_score(profile, requirements)
        │     seniority_score  = compute_seniority_score(profile, requirements)
        │
        ├── Phase 4: LLM narrates the Python-computed scores
        │     self.assess(profile, requirements, matched_required,
        │                 missing_required, skills_score, experience_score,
        │                 seniority_score, dates_extracted_ok)
        │     └─► _AssessRationale { rationale }
        │     assessment = FitAssessment(... all scores Python-set ...
        │                                 strengths = matched_required ∪ matched_nice,
        │                                 gaps      = missing_required, ...)
        │     assessment.overall_score = combine_subscores(assessment)
        │
        └── Phase 5: score-calibrated advice
              with dspy.context(lm=creative_lm):
                  self.advise(assessment, overall_score, target_role)
                  └─► list[str]   (3-5 suggestions)

   Final: dspy.Prediction(profile, assessment, suggestions)
```

## File-by-file

### `config.py`

Loads `GROQ_API_KEY` from `.env` via `python-dotenv`. Single source of truth for the API key.

### `resume_analyser.py`

| Section | What it does |
|---|---|
| LM setup | `precise_lm` (`llama-3.3-70b-versatile`, temp 0.0) is the global default via `dspy.configure(lm=precise_lm)` — used for JD parse, profile extraction, rationale narration. `creative_lm` (`llama-3.1-8b-instant`, temp 0.2) overrides via `dspy.context` for `GenerateAdvice` only. The split keeps a 10-candidate run inside Groq's 100K TPD cap on 70B |
| Pydantic schemas | `WorkExperience`, `Project`, `Certification`, `ResumeProfile`, `JobRequirements`, `_AssessRationale`, `FitAssessment`. Schemas are both `OutputField` types and the input contract for downstream stages |
| Skill evidence | `cert_knowledge.json` loader/saver, `classify_cert`, `cert_weight`, `build_evidence_map`, and the `EV_*` weight constants — the evidence layer between matching and scoring |
| Signatures | `ParseJobDescription`, `ExtractProfile`, `AssessFit`, `GenerateAdvice`, plus the knowledge-resolvers `ResolveSkillEquivalences` and `ClassifyCertification`. The docstring of each is the instruction passed to the model |
| Skill matching | `skill_knowledge.json` (LLM-resolved equivalences) + `resolve_skill`, `skill_match_set`, `match_skills`, `_expand_candidate_stack`, `candidate_stack`. `SYNONYMS` remains as the offline fallback |
| Date math | `_parse_ym`, `compute_years_experience`, `reconcile_years`, `dates_present` — `dateutil` for fuzzy/partial dates, merge-intervals for years, structural checks for hallucinated dates |
| Extraction post-processing | `_normalize`, `_contains`, `_word_in` (string helpers); `recover_missed_skills`, `recover_email`, `filter_hallucinated_extraction` |
| Claimed-years regex | `_YEARS_PATTERNS` + `extract_claimed_years` — folded into the Extraction post-processing section since it operates on `resume_text` |
| Cosmetic cleanup | `clean_name` (kerning repair), `drop_empty_roles`, `filter_education` (`_DEGREE_KEYWORDS`), `drop_non_certifications` (role-vs-credential guard) |
| Sub-scores + combination | Dedicated "Scoring" section: `compute_relevant_years`, `compute_skills_match_score`, `compute_experience_match_score`, seniority block (`_SENIORITY_RANK`, `_RANK_TITLE_KEYWORDS`, `_infer_candidate_rank`, `compute_seniority_score`), `combine_subscores` (50/30/20 weighted sum) |
| Modules | `JDParser` and `ResumeAnalyser` — the orchestration. `ResumeAnalyser.forward` runs the five phases in order |

### `test_analyser.py`

| Section | What it does |
|---|---|
| Constants | `ROOT`, `JD_DIR`, `PROFILES_DIR`, `CACHE_PATH`, `RESULTS_PATH`, `JD_COOLDOWN_S`, `CANDIDATE_PACING_S`, `JD_TO_FOLDER` |
| `call_with_retry` | Catches Groq 429s, parses `"try again in Xs"`, sleeps that long + 3s buffer, bounded retries, re-raises last exception explicitly |
| `read_pdf` | One-liner over `fitz.open(...).get_text()` |
| `save_results` | Writes `analyser_results.json` after every candidate — partial runs are preserved if the process dies |
| `get_or_parse_requirements` | Returns cached `JobRequirements` if `requirements_cache.json` has an entry; else parses via LLM and writes back |
| `run_one` | Single-candidate runner — prints results to stdout and returns a dict for the JSON output |
| `main` | Iterates JDs × candidate folders, paces between candidates, saves incrementally |

### `requirements_cache.json` (auto-generated)

Map of `{jd_filename: JobRequirements.model_dump()}`. Lives at the project root. Delete it to force re-parsing on the next run.

### `skill_knowledge.json` (seed + auto-grown)

Keyed by normalized canonical skill → `{aliases, entailed_by, related_only, confidence, source}`. Hand-seeded entries (`source: "seed"`) anchor error-prone cases (Docker/Kubernetes/Java); the rest are filled by the 70B resolver (`source: "llm-70b"`) once per skill and cached. Delete an entry to force re-resolution.

### `cert_knowledge.json` (seed + auto-grown)

`_patterns` (official-source rules) + learned instances `{tier, reputation, skills, validity_years, …}`. Same lifecycle as the skill KB.

### `analyser_results.json` (auto-generated)

Array of per-candidate result objects (or error stubs if a candidate failed). Written after each candidate, so partial runs are preserved.

## Where DSPy stops and Python takes over

| Task | DSPy | Python |
|---|:---:|:---:|
| Read PDF | | ✓ |
| Parse JD into requirements | ✓ | |
| Cache parsed JDs across runs | | ✓ |
| Extract structure (work history, skills lists) | ✓ | |
| Verify every extracted item is grounded in source text | | ✓ |
| Recover skills the LLM dropped | | ✓ |
| Recover email lost to glyph artefacts | | ✓ |
| Enforce stack ⊆ technical_skills | | ✓ |
| Parse a single date string | | ✓ |
| Compute total tenure across multiple roles | | ✓ |
| Detect failed or fabricated date extraction | | ✓ |
| Pick between computed vs claimed years | | ✓ |
| Compute years-of-*relevant*-experience | | ✓ |
| Extract claimed-years figure from resume header | | ✓ |
| Resolve a skill's equivalences (one-time, cached) | ✓ | |
| Match candidate skills against JD required/nice (apply the knowledge) | | ✓ |
| Score each skill by evidence (project/role/cert/listed) | | ✓ |
| Classify a KNOWN cert (cache / official `_patterns` / heuristic) | | ✓ |
| Classify an UNKNOWN cert (one-time enrichment, then cached) | ✓ | |
| Compute skills / experience / seniority sub-scores | | ✓ |
| Produce rationale | ✓ | |
| Combine sub-scores into overall_score | | ✓ |
| Generate score-calibrated suggestions | ✓ | |
| Validate output types | DSPy + Pydantic | |
| Handle rate limits / retry | | ✓ |
| Persist results | | ✓ |

**The general principle: DSPy for natural-language judgement, Python for everything else.** Earlier versions let the LLM compute years-of-experience, do the skill matching, and choose the overall score directly — each produced bugs that disappeared the moment Python took over. The current version is the result of pushing this principle as far as it goes without sacrificing the parts where the LLM is genuinely useful (interpreting prose, writing suggestions, ratio-based scoring with rationale).

## Things this architecture deliberately does *not* do (and why)

- **No prompt files.** Instructions live as docstrings on the four Signature classes. If they need to be optimised, DSPy's MIPROv2 or similar would refine them in-place. Externalising prompts to a `.md` file would lose DSPy's granular control and break the optimisation path.
- **No `dspy.Refine` retry loop yet.** Hallucination filtering is currently a silent drop. The clean next step is wrapping `ExtractProfile` in `dspy.Refine` with a `reward_fn` that asserts every extracted item is in `resume_text` — converts silent drops to a feedback loop. See [DSPy assertions docs](https://dspy.ai/learn/programming/7-assertions/).
- **No external skill ontology yet.** `SYNONYMS` is hand-curated and covers the fixture. For production use, swap it for the EMSI/Lightcast skill DB bundled by [SkillNER](https://github.com/AnasAito/SkillNER) (~34k skills with synonyms) — same API, much wider coverage.
- **No few-shot bootstrapping.** `BootstrapFewShot` needs a metric and a labelled set. Premature here; revisit once 15+ hand-labelled (resume, JD, expected sub-scores) triples exist.
- **No ranking step.** Each candidate is scored independently. To produce a ranked shortlist per JD, sort `analyser_results.json` by `assessment.overall_score` — one-liner, not worth a stage.
