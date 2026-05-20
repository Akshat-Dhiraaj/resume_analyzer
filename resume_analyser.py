"""Resume analyser — DSPy 3.x + Groq (70B precise / 8B creative).

Hybrid LLM/Python pipeline. The LLM handles natural-language judgement:
JD parsing, profile extraction, rationale narration, and suggestion writing.
Everything with a deterministic answer is computed in Python:

  - synonym-aware skill matching (set arithmetic with curated SYNONYMS)
  - year math (merge-intervals over work history)
  - seniority alignment (title rank vs JD-target rank)
  - claimed-years (regex over the resume header)
  - hallucination filters (verbatim + strict-subset against source text)
  - skill recovery (lexicon scan to backfill items the LLM missed)
  - overall_score (weighted sum of sub-scores)

The deterministic passes are defence-in-depth against any LLM drift —
they're additive, not punitive, so they stay in place regardless of model
size. Skill recovery backfills items the LLM missed; the verbatim filter
drops items that aren't in the source; the strict-subset rule keeps a
role's stack honest about what the candidate claims as a skill.
"""

import datetime
import re
import dspy
from dateutil import parser as dateparser
from pydantic import BaseModel, Field
from config import GROQ_API_KEY

# ─────────────────────────────────────────────
# LM setup — 70B for accuracy, 8B for advice writing
# ─────────────────────────────────────────────
# precise_lm (70B, temp 0.0) is the default for stages where accuracy and
# reproducibility matter: JD parse, profile extraction, rationale narration.
# creative_lm (8B, temp 0.2) is used only for GenerateAdvice — a cheap
# generative task that doesn't justify 70B tokens, and where mild variability
# is desirable. With this split, a 10-candidate run fits inside Groq's
# free-tier 100K-TPD cap on 70B.
precise_lm = dspy.LM(
    "groq/llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY, max_tokens=2048, temperature=0.0,
)
creative_lm = dspy.LM(
    "groq/llama-3.1-8b-instant",
    api_key=GROQ_API_KEY, max_tokens=2048, temperature=0.2,
)
dspy.configure(lm=precise_lm)

# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────
class WorkExperience(BaseModel):
    role: str
    company: str
    start_date: str = Field(description="YYYY-MM or YYYY format. Empty if unknown.")
    end_date: str = Field(description="YYYY-MM, YYYY, or 'present'. Empty if unknown.")
    stack: list[str] = Field(
        default_factory=list,
        description="Technologies named in THIS role's bullets. A downstream "
                    "Python pass enforces stack ⊆ technical_skills so the LLM "
                    "can't graft popular tokens onto every role.",
    )

class ResumeProfile(BaseModel):
    name: str
    email: str | None = None
    work_history: list[WorkExperience]
    summary_claimed_years: float = Field(
        default=0.0, ge=0,
        description="Python-extracted via regex from the resume header — "
                    "leave at 0.0, the LLM does not need to populate this.",
    )
    # Computed in Python — DO NOT extract via LLM:
    years_experience: float = Field(default=0.0, ge=0)
    years_relevant_experience: float = Field(default=0.0, ge=0)
    dates_extracted_ok: bool = Field(default=True)
    technical_skills: list[str]
    soft_skills: list[str]
    education: list[str]

class JobRequirements(BaseModel):
    role_title: str
    seniority: str = Field(description="One of: junior, mid, senior, staff, principal, unknown")
    min_years_experience: float = Field(default=0.0, ge=0)
    required_skills: list[str] = Field(
        description="Skills the JD marks as mandatory, must-have, core, or "
                    "'you have' (vs 'plus'/'bonus'). Be canonical: 'AWS' not "
                    "'experience with Amazon Web Services'. Group obvious "
                    "synonyms under one entry.",
    )
    nice_to_have_skills: list[str] = Field(
        description="Skills mentioned as 'plus', 'bonus', 'preferred', "
                    "'good to have', or in non-mandatory sections.",
    )
    responsibilities: list[str] = Field(
        description="Day-to-day responsibilities described in the JD."
    )

class FitAssessment(BaseModel):
    """Final assessment object.

    Every numeric score and every list field is Python-populated in
    ResumeAnalyser.forward. The LLM (via _AssessRationale) only writes the
    `rationale` string, which narrates the sub-scores."""
    skills_match_score: int = Field(default=0, ge=0, le=100)
    experience_match_score: int = Field(default=0, ge=0, le=100)
    seniority_match_score: int = Field(default=0, ge=0, le=100)
    # Python-computed via curated synonym matching (deterministic):
    matched_required: list[str] = Field(default_factory=list)
    missing_required: list[str] = Field(default_factory=list)
    matched_nice_to_have: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    rationale: str = ""
    overall_score: int = Field(default=0, ge=0, le=100)

class _AssessRationale(BaseModel):
    """LLM output for AssessFit — rationale only.

    Every sub-score is computed in Python now (the LLM was inflating ratios
    even when given Python-provided ground truth: 5/8 matched → 85, etc.).
    The LLM's job is reduced to narrating the numbers Python produced."""
    rationale: str = Field(description="2-3 sentences explaining the sub-scores.")

# ─────────────────────────────────────────────
# Signatures
# ─────────────────────────────────────────────
class ParseJobDescription(dspy.Signature):
    """Parse a raw job description into structured requirements.

    Distinguish required from nice-to-have by language: words like 'must',
    'required', 'core', 'mandatory', 'you have' → required_skills.
    Words like 'plus', 'bonus', 'preferred', 'nice to have', 'good to have'
    → nice_to_have_skills.

    Be canonical with skill names: 'AWS' not 'Amazon Web Services',
    'CI/CD' not 'continuous integration and continuous deployment'.
    Group obvious synonyms under one entry."""
    jd_text: str = dspy.InputField()
    requirements: JobRequirements = dspy.OutputField()

class ExtractProfile(dspy.Signature):
    """Extract a structured profile from raw resume text.

    GROUND RULE: every value you emit must appear VERBATIM somewhere in
    resume_text. Never invent, infer, or copy-from-elsewhere. If a piece
    of information is not present, leave the field empty/null/0.0. A
    downstream Python pass will discard any item not found in resume_text,
    so inventing wastes tokens.

    For work_history:
      - Include only PAID PROFESSIONAL ROLES with an employer name.
      - Use the dates exactly as shown in the resume. YYYY-MM if month and
        year are given, YYYY if only year. For ongoing roles, end_date =
        'present'. If a role has NO dates in the resume, leave start_date
        and end_date as empty strings — DO NOT guess, fabricate, or copy
        dates from an adjacent role. A flag downstream will surface this.
      - For each role, populate `stack` with technologies named in THAT
        role's bullet points or description. Do NOT copy from the
        candidate's overall technical_skills section — if the role's
        bullets only mention "Java, Spring Boot", the stack is exactly
        ["Java", "Spring Boot"], even if the candidate lists React
        elsewhere on the resume.
      - DO NOT include: personal projects, hackathons, course projects.

    For technical_skills:
      - Copy the candidate's skills section verbatim. Preserve sub-items
        like "Functional Testing", "Regression Testing" — do NOT collapse
        them into a category header. If the section lists 20 items, return
        20 items.

    For education:
      - Include only post-secondary entries (B.Tech, B.E., B.Sc., M.Sc.,
        M.Tech, MBA, PhD, diploma). Skip 10th/12th/school entries.

    For summary_claimed_years: leave at 0.0 — Python will extract this
    from resume_text via regex; you don't need to.

    For years_experience and years_relevant_experience: leave at 0.0 — both
    are computed deterministically in Python from work_history."""
    resume_text: str = dspy.InputField()
    profile: ResumeProfile = dspy.OutputField()

class AssessFit(dspy.Signature):
    """Write a 2-3 sentence rationale explaining the candidate's fit.

    All sub-scores and the matched/missing lists are PROVIDED to you (computed
    deterministically in Python). You do not produce numbers — you explain
    them. Reference specific matched skills and specific gaps. If
    dates_extracted_ok is False, the rationale MUST acknowledge that the
    candidate's experience figure is unverified."""
    profile: ResumeProfile = dspy.InputField()
    requirements: JobRequirements = dspy.InputField()
    matched_required: list[str] = dspy.InputField()
    missing_required: list[str] = dspy.InputField()
    skills_match_score: int = dspy.InputField()
    experience_match_score: int = dspy.InputField()
    seniority_match_score: int = dspy.InputField()
    dates_extracted_ok: bool = dspy.InputField()
    out: _AssessRationale = dspy.OutputField()

class GenerateAdvice(dspy.Signature):
    """Produce 3-5 concrete, actionable resume improvements, calibrated to
    the candidate's overall_score:

      - overall_score < 50 (poor fit):
          The first suggestion MUST honestly flag the stack/role mismatch and
          recommend either pivoting toward a better-aligned role family, or
          undertaking specific learning to bridge the gap. Don't sugarcoat.
      - 50 <= overall_score < 70 (moderate fit):
          Suggest filling specific high-impact gaps with concrete projects
          or sections.
      - overall_score >= 70 (strong fit):
          Suggest polish — quantify impact, add metrics, surface less-visible
          relevant work, improve keyword density for ATS.

    Each suggestion must reference a specific gap from the assessment and
    propose a specific change. No generic advice."""
    assessment: FitAssessment = dspy.InputField()
    overall_score: int = dspy.InputField()
    target_role: str = dspy.InputField()
    suggestions: list[str] = dspy.OutputField()

# ─────────────────────────────────────────────
# Deterministic skill matching (synonym-aware)
# ─────────────────────────────────────────────
SYNONYMS: dict[str, list[str]] = {
    # web / MERN
    "HTML5":            ["html", "html5"],
    "CSS3":             ["css", "css3", "sass", "scss", "tailwind css", "tailwind"],
    "JavaScript":       ["javascript", "js", "ecmascript", "es6", "es6+", "javascript (es6+)"],
    "TypeScript":       ["typescript", "ts"],
    "Node.js":          ["node.js", "nodejs", "node"],
    "Express.js":       ["express.js", "express", "expressjs", "express js"],
    "React.js":         ["react.js", "react", "reactjs", "react 18", "react hooks", "next.js", "nextjs"],
    "MongoDB":          ["mongodb", "mongo", "mongoose"],
    "RESTful APIs":     ["restful apis", "rest apis", "rest api", "restful", "rest assured", "rest"],
    "AWS":              ["aws", "amazon web services", "ec2", "s3", "iam", "aws secrets manager", "aws (ec2, s3, secrets manager, iam)"],
    "Docker":           ["docker", "kubernetes", "k8s"],
    "CI/CD":            ["ci/cd", "ci-cd", "cicd", "jenkins", "gitlab ci", "github actions", "bitbucket pipelines", "ci/cd pipelines", "ci/cd pipeline"],
    "Git":              ["git", "github", "gitlab", "bitbucket"],
    "GraphQL":          ["graphql"],
    "Agile":            ["agile", "agile/scrum", "agile development", "scrum", "kanban"],
    "Scrum":            ["scrum", "agile/scrum", "agile"],
    # QA
    "Selenium":         ["selenium", "selenium webdriver"],
    "JMeter":           ["jmeter", "apache jmeter"],
    "TestNG":           ["testng"],
    "Postman":          ["postman"],
    "REST Assured":     ["rest assured", "restassured"],
    "Gatling":          ["gatling"],
    "LoadRunner":       ["loadrunner", "load runner"],
    "Performance Testing": ["performance testing", "load testing", "jmeter", "gatling", "loadrunner", "stress testing"],
    "API Testing":      ["api testing", "postman", "rest assured", "swagger", "openapi"],
    "Manual Testing":   ["manual testing", "functional testing", "regression testing", "smoke testing", "sanity testing", "uat", "user acceptance testing", "exploratory testing", "cross-browser testing"],
    "Test Automation":  ["test automation", "automation testing", "selenium", "selenium webdriver", "testng", "cypress", "playwright", "appium"],
    # languages
    "Python":           ["python"],
    "Java":             ["java"],
    # AI/ML
    "PyTorch":          ["pytorch", "torch"],
    "CNNs":             ["cnn", "cnns", "convolutional neural network", "convolutional neural networks",
                          "resnet", "vgg", "yolo", "yolov5", "yolov8", "efficientnet", "mobilenet",
                          "unet", "u-net"],
    "Transformers":     ["transformer", "transformers", "transformer based", "transformer-based",
                          "transformer-based architectures", "vision transformer", "vit",
                          "bert", "gpt", "llama", "llama 3", "mistral", "qwen", "whisper", "llm", "llms"],
    "Hugging Face Transformers": ["hugging face", "huggingface", "hf transformers",
                          "huggingface transformers", "hf"],
    "Machine Learning": ["machine learning", "ml", "scikit-learn", "sklearn",
                          "tensorflow", "tensorflow keras", "keras", "pytorch", "xgboost",
                          "lightgbm", "random forest"],
    "Computer Vision":  ["computer vision", "cv", "opencv", "image processing", "yolo",
                          "paddleocr", "ocr"],
    "Deep Learning":    ["deep learning", "dl", "neural networks", "tensorflow", "keras",
                          "pytorch", "rnn", "lstm", "gru"],
    "FastAPI":          ["fastapi", "fast api"],
    "DICOM":            ["dicom", "pydicom"],
    "LangChain":        ["langchain", "langgraph"],
    "RAG systems":      ["rag", "rag systems", "rag pipeline", "retrieval augmented generation", "retrieval-augmented generation"],
    "Vector databases": ["vector db", "vector database", "vector databases", "pinecone", "weaviate", "chromadb", "qdrant", "faiss", "milvus"],
}

# Filter pieces produced by splitting that are too generic to safely match.
_GENERIC_TOKENS = {"and", "or", "with", "of", "the", "for", "in", "on", "to", "a", "an"}

def _expand_candidate_stack(skills: set[str]) -> set[str]:
    """Split compound skills like 'AWS (EC2, S3, Secrets Manager)' into atoms.

    Returns a lowercased set containing both the original phrases and their
    parenthesised / comma-separated pieces. Used to make synonym matching
    forgiving of resume formatting without resorting to substring search."""
    out: set[str] = set()
    for s in skills:
        s = s.lower().strip()
        if not s:
            continue
        out.add(s)
        for piece in re.split(r"[(),&/\-]", s):
            piece = piece.strip()
            if piece and piece not in _GENERIC_TOKENS:
                out.add(piece)
    return out

def candidate_stack(profile: "ResumeProfile") -> set[str]:
    """All skills the candidate has demonstrated, across technical_skills
    and per-role work-history stack entries."""
    stack: set[str] = set(profile.technical_skills)
    for w in profile.work_history:
        stack.update(w.stack)
    return stack

def match_skills(stack: set[str], required: list[str]) -> tuple[list[str], list[str]]:
    """Return (matched, missing) where matching uses the curated SYNONYMS map.

    A required skill matches if any of its aliases (lowercased) appears as an
    exact element of the candidate's expanded stack. Preserves the JD's
    canonical wording in the returned lists."""
    expanded = _expand_candidate_stack(stack)
    matched, missing = [], []
    for req in required:
        aliases = {a.lower() for a in SYNONYMS.get(req, [req])} | {req.lower()}
        if aliases & expanded:
            matched.append(req)
        else:
            missing.append(req)
    return matched, missing

# ─────────────────────────────────────────────
# Deterministic date math
# ─────────────────────────────────────────────
_PRESENT_TOKENS = {"present", "current", "now", "ongoing", "till date", "to date"}

def _parse_ym(s: str, today: datetime.date) -> datetime.date | None:
    if not s or not s.strip():
        return None
    s = s.strip().lower()
    if s in _PRESENT_TOKENS:
        return today
    try:
        dt = dateparser.parse(s, default=datetime.datetime(2000, 1, 1), fuzzy=True)
        return dt.date()
    except (ValueError, OverflowError):
        return None

def compute_years_experience(work_history: list[WorkExperience], today: datetime.date) -> float:
    """Sum non-overlapping work durations, in years (1 decimal)."""
    intervals: list[tuple[datetime.date, datetime.date]] = []
    for w in work_history:
        s = _parse_ym(w.start_date, today)
        e = _parse_ym(w.end_date, today) or today
        if s and e > s:
            intervals.append((s, e))
    intervals.sort()
    merged: list[tuple[datetime.date, datetime.date]] = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    days = sum((e - s).days for s, e in merged)
    return round(days / 365.25, 1)

def reconcile_years(computed: float, claimed: float) -> float:
    if claimed <= 0: return computed
    if computed <= 0: return claimed
    if computed > claimed + 2.0: return claimed
    return computed

def dates_present(work_history: list[WorkExperience]) -> bool:
    """True if work-history dates look genuine.

    No work history at all → False.
    Zero roles with a usable start_date → False.
    >1 role, ALL with non-empty starts AND all sharing the same start_date
      → False (LLM almost certainly hallucinated a single year and copied it
      across; observed for Abhilash whose 4 roles all came back as
      '2018' → 'present').
    >1 role, ALL with non-empty starts AND all ending in 'present'
      → False (only one role can realistically be ongoing).
    Otherwise True — partial date data is OK (e.g. Shiv has one role with
    real dates and one open-source contribution with empty dates)."""
    if not work_history:
        return False
    starts = [w.start_date.strip() for w in work_history if w.start_date and w.start_date.strip()]
    if not starts:
        return False
    all_have_starts = len(starts) == len(work_history)
    if len(work_history) > 1 and all_have_starts:
        if len({s.lower() for s in starts}) == 1:
            return False
        ends = [w.end_date.strip().lower() for w in work_history]
        if all(e in _PRESENT_TOKENS for e in ends):
            return False
    return True

# ─────────────────────────────────────────────
# Extraction post-processing (source-text verification)
# ─────────────────────────────────────────────
def _normalize(s: str) -> str:
    """Lowercase + collapse all non-alphanumeric to single spaces.
    Handles kerning artefacts ("R jztshiva") and punctuation variation
    when checking 'is this string in the resume?'."""
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

def _contains(needle: str, haystack_norm: str) -> bool:
    n = _normalize(needle)
    return bool(n) and n in haystack_norm

def _word_in(needle: str, haystack_lower: str) -> bool:
    """Word-boundary substring check. Treats '.' as part of a token so
    'node.js' doesn't accidentally match inside 'node.json'. Avoids the
    classic 'ml' in 'html' false positive."""
    needle = needle.lower().strip()
    if not needle:
        return False
    pattern = r"(?<![a-z0-9.])" + re.escape(needle) + r"(?![a-z0-9.])"
    return bool(re.search(pattern, haystack_lower))

def recover_missed_skills(profile: "ResumeProfile", resume_text: str) -> "ResumeProfile":
    """Recover skills the 8B extractor dropped.

    8B is known to under-extract on noisy/kerned PDFs — Divya's skills
    collapsed to section headers, Shivkumar's email was lost to a glyph
    artefact, etc. This pass scans resume_text against the curated SYNONYMS
    lexicon (using word-boundary matching) and adds any canonical skill
    whose alias appears in the source but isn't already in technical_skills.

    Net effect: the LLM's job becomes 'extract structure', and the
    deterministic pass handles 'recover the flat skill inventory'."""
    src_lower = resume_text.lower()
    existing_norm = {_normalize(s) for s in profile.technical_skills}

    for canonical, aliases in SYNONYMS.items():
        if _normalize(canonical) in existing_norm:
            continue
        for alias in [canonical, *aliases]:
            if _word_in(alias, src_lower):
                profile.technical_skills.append(canonical)
                existing_norm.add(_normalize(canonical))
                break
    return profile

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

def recover_email(profile: "ResumeProfile", resume_text: str) -> "ResumeProfile":
    """If the LLM lost the email (8B has dropped this on glyph-corrupted PDFs
    like Shivkumar's 'R jztshiva@...'), recover via regex on the source.
    Picks the first plausible address. Leaves profile.email alone if set."""
    if profile.email and "@" in profile.email:
        return profile
    m = _EMAIL_RE.search(resume_text)
    if m:
        profile.email = m.group(0)
    return profile

def _bare(s: str) -> str:
    """Strip parenthetical detail then normalize. 'AWS (EC2, S3, IAM)' → 'aws',
    'Python (Basic)' → 'python'. Lets dedupe collapse canonical-vs-detail
    variants where one entry's bare form equals another's normalized form."""
    return _normalize(re.sub(r"\(.*?\)", "", s))

def _dedupe(items: list[str]) -> list[str]:
    """Order-preserving deduplication. Skips an entry if either its full
    normalized form OR its paren-stripped bare form has already been seen.

    Examples:
      ['Spring Boot', 'Spring Cloud', 'Spring Boot']   → drops the dup
      ['Javascript', 'TypeScript', 'JavaScript']       → case-insensitive
      ['AWS (EC2, S3, IAM)', 'AWS']                    → drops bare canonical
      ['Spring', 'Spring Boot']                        → KEEPS both (distinct
                                                          products, not paren-variant)

    The 70B extractor emits the same skill twice within work_history[].stack,
    and recovery appends canonical names that overlap with detail-bearing
    variants the LLM already extracted. Both classes get collapsed here."""
    seen: set[str] = set()
    out: list[str] = []
    for s in items:
        key  = _normalize(s)
        bare = _bare(s)
        if not key or key in seen or bare in seen:
            continue
        seen.add(key)
        if bare:
            seen.add(bare)
        out.append(s)
    return out

def filter_hallucinated_extraction(profile: "ResumeProfile", resume_text: str) -> "ResumeProfile":
    """Three-pass cleanup of LLM extraction:

    Pass 1 (verbatim): drop any skill/stack entry not found anywhere in
    resume_text. Catches fully-invented tokens.

    Pass 2 (strict subset for work_history.stack): the per-role stack must
    be a subset of the candidate's overall technical_skills. The LLM tends
    to graft popular tokens (React, Node.js, Docker) onto every role's stack
    even when the candidate doesn't list them as a skill — this caused
    Sarthak (Java/Spring/Angular at IDEMIA) to be scored as a MERN dev.

    Pass 3 (dedupe): collapse case-insensitive duplicates in both
    technical_skills and each role's stack. Keeps the first occurrence."""
    src = _normalize(resume_text)
    profile.technical_skills = _dedupe([s for s in profile.technical_skills if _contains(s, src)])
    profile.soft_skills      = _dedupe([s for s in profile.soft_skills      if _contains(s, src)])

    skill_set = {_normalize(s) for s in profile.technical_skills}
    for w in profile.work_history:
        w.stack = _dedupe([
            s for s in w.stack
            if _contains(s, src) and _normalize(s) in skill_set
        ])
    return profile

# Claimed-years extraction (continues the source-text verification theme)
_YEARS_PATTERNS = [
    # Naukri header tag: '3y 1m' / '3y_1m' / '3y-1m'
    re.compile(r"\b(\d+)\s*y(?:r|rs|ear|ears)?\s*[_\-\s]?\s*(\d+)\s*m(?:o|os|on|onth|onths)?\b", re.I),
    # 'X+ years' / 'X years experience'
    re.compile(r"\b(\d+(?:\.\d+)?)\s*\+?\s*y(?:r|rs|ear|ears)?\s*(?:of\s+)?(?:experience|exp)\b", re.I),
    # bare 'X years' (less specific, only first 400 chars)
    re.compile(r"\b(\d+(?:\.\d+)?)\s*\+?\s*years?\b", re.I),
]

def extract_claimed_years(resume_text: str) -> float:
    """Pull a self-stated experience claim from the resume header.

    Returns the highest value found in the first ~600 chars. 0.0 if no
    explicit claim. Handles Naukri's 'Xy Ym' tag and the more common
    prose 'X+ years experience' summary opener."""
    head = resume_text[:600]
    candidates: list[float] = []
    m = _YEARS_PATTERNS[0].search(head)
    if m:
        candidates.append(round(int(m.group(1)) + int(m.group(2)) / 12, 1))
    for p in _YEARS_PATTERNS[1:]:
        for hit in p.finditer(head):
            try:
                candidates.append(float(hit.group(1)))
            except ValueError:
                pass
    return max(candidates) if candidates else 0.0

# ─────────────────────────────────────────────
# Cosmetic post-processing (presentation cleanup, no scoring impact)
# ─────────────────────────────────────────────
_DEGREE_KEYWORDS = (
    "b.tech", "btech", "b.e.", "be.", "b.sc", "bsc", "bachelor",
    "m.tech", "mtech", "m.sc", "msc", "master", "mba",
    "phd", "ph.d", "doctorate", "diploma", "engineering",
)

def filter_education(entries: list[str]) -> list[str]:
    """Keep only post-secondary entries. Drops 10th/12th/school lines that
    the LLM sometimes copies in (Tarun's 'SHRI BHAWANI NIKETAIN PUBLIC SCHOOL')."""
    return [e for e in entries if any(kw in e.lower() for kw in _DEGREE_KEYWORDS)]

def clean_name(name: str) -> str:
    """Repair kerning artefacts in extracted names.

    PDFs sometimes render 'ABHISHEK MAJHI' as 'A B H I S H E K  M A J H I'
    (single chars separated by spaces, words separated by double spaces).
    Detect by counting single-char tokens; if ≥60% are single chars, collapse
    intra-word spaces and preserve inter-word boundaries (the double space)."""
    name = (name or "").strip()
    parts = name.split()
    if not parts:
        return name
    single_chars = sum(1 for p in parts if len(p) == 1)
    if single_chars / len(parts) >= 0.6:
        return " ".join(p.replace(" ", "") for p in re.split(r"\s{2,}", name))
    return name

def drop_empty_roles(work_history: list[WorkExperience]) -> list[WorkExperience]:
    """Drop entries with empty role or company. The LLM occasionally promotes
    a non-role line (open-source projects, course completions) into a
    work_history entry with a blank role string."""
    return [w for w in work_history if w.role.strip() and w.company.strip()]

# ─────────────────────────────────────────────
# Scoring (Python-computed sub-scores + overall combiner)
# ─────────────────────────────────────────────
def compute_relevant_years(years_experience: float, matched_count: int, required_count: int) -> float:
    """Total years × (matched / required) ratio.

    Takes pre-computed match counts so the caller doesn't recompute
    `match_skills` (which is already needed for the assessment). If no
    required skills are specified, returns total years unchanged.

    Nice-to-have skills are NOT included: they signal fit-bonus, not
    relevance. Penalising an in-family candidate's relevant years for
    missing optional skills was the bug that crushed Divya's score from
    4.3y to 1.3y in the early version."""
    if required_count == 0:
        return years_experience
    return round(years_experience * matched_count / required_count, 1)

def compute_skills_match_score(matched: list[str], required: list[str]) -> int:
    """Ratio of matched to required, rounded to integer. 80 if no requirements
    are specified (defensive). Replaces the LLM sub-score, which inflated
    consistently (Aakash 5/8 → 85, Divya 8/14 → 85, etc.)."""
    if not required:
        return 80
    return round(100 * len(matched) / len(required))

def compute_experience_match_score(profile: "ResumeProfile", requirements: JobRequirements) -> int:
    """Years-of-relevant-experience vs min-required, with date-verification cap.

    If min_years_experience > 0:  ratio × 100, capped at 100.
    If min_years_experience == 0: ladder on absolute years_relevant_experience
      (<1y→40, 1y→60, 2y→75, 3y→90, 5y+→100).
    If profile.dates_extracted_ok is False, the final score is capped at 50."""
    rel = profile.years_relevant_experience
    min_y = requirements.min_years_experience
    if min_y > 0:
        score = min(round(100 * rel / min_y), 100)
    elif rel < 1:   score = 40
    elif rel < 2:   score = 60
    elif rel < 3:   score = 75
    elif rel < 5:   score = 90
    else:           score = 100
    if not profile.dates_extracted_ok:
        score = min(score, 50)
    return score

_SENIORITY_RANK = {
    "junior": 1, "entry": 1, "associate": 1,
    "mid": 2, "intermediate": 2, "unknown": 2,
    "senior": 3, "lead": 3,
    "staff": 4,
    "principal": 5, "distinguished": 5,
}

# (rank, title-substring-keywords) — first match wins, highest-rank groups first.
_RANK_TITLE_KEYWORDS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (5, ("principal", "distinguished")),
    (4, ("staff",)),
    (3, ("senior", "lead", "sr.")),
    (1, ("junior", "intern", "trainee", "associate")),
)

def _infer_candidate_rank(profile: ResumeProfile) -> int:
    """Best guess at the candidate's current seniority from role titles.
    Default 2 (mid) when titles are ambiguous ('Software Engineer', etc.)."""
    titles = " ".join(w.role.lower() for w in profile.work_history)
    for rank, keywords in _RANK_TITLE_KEYWORDS:
        if any(kw in titles for kw in keywords):
            return rank
    return 2

def compute_seniority_score(
    profile: ResumeProfile,
    requirements: JobRequirements,
) -> int:
    """Seniority alignment score, computed deterministically.

    Replaces the LLM sub-score. The LLM was unreliable here — it under-scored
    Vishal (Senior→Mid: 70, expected 85-95) and over-scored Shiv (Mid at
    target with years < min: 85, expected 60-75) in the latest run."""
    target = _SENIORITY_RANK.get(requirements.seniority.lower().strip(), 2)
    cand   = _infer_candidate_rank(profile)
    diff   = cand - target
    meets_years = profile.years_experience >= max(requirements.min_years_experience, 0.0)

    if diff == 0:
        return 85 if meets_years else 68
    if diff == 1:
        return 90                          # one above target — over-qualified but adjacent
    if diff == -1:
        return 65                          # one below target — stretch
    if diff >= 2:
        return 70                          # significantly senior of target — usually fine
    return 35                              # two+ levels under target

def combine_subscores(a: FitAssessment) -> int:
    """Weighted sum: skills 50%, experience 30%, seniority 20%."""
    return round(0.5 * a.skills_match_score
               + 0.3 * a.experience_match_score
               + 0.2 * a.seniority_match_score)

# ─────────────────────────────────────────────
# Modules
# ─────────────────────────────────────────────
class JDParser(dspy.Module):
    """Parses raw JD text into structured requirements. Cache externally
    (per JD) so we don't re-parse for every candidate."""
    def __init__(self):
        super().__init__()
        self.parse = dspy.ChainOfThought(ParseJobDescription)

    def forward(self, jd_text: str) -> JobRequirements:
        return self.parse(jd_text=jd_text).requirements

class ResumeAnalyser(dspy.Module):
    """Five-phase pipeline: LLM extract → Python clean/recover/match/score
    → LLM rationale → LLM advice. The LLM does no arithmetic; every numeric
    field on FitAssessment is Python-computed before AssessFit sees it.

    See ARCHITECTURE.md for the per-phase responsibility map."""
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(ExtractProfile)
        self.assess  = dspy.ChainOfThought(AssessFit)
        self.advise  = dspy.ChainOfThought(GenerateAdvice)

    def forward(
        self,
        resume_text: str,
        requirements: JobRequirements,
        target_role: str,
    ):
        # ── Phase 1: extract + clean ───────────────────────────────────────
        # Recovery must precede the filters so that lexicon-recovered skills
        # are available to back legitimate work_history.stack entries when
        # the strict-subset rule runs.
        profile = self.extract(resume_text=resume_text).profile
        profile.name         = clean_name(profile.name)
        profile.work_history = drop_empty_roles(profile.work_history)
        profile.education    = filter_education(profile.education)
        profile = recover_missed_skills(profile, resume_text)
        profile = recover_email(profile, resume_text)
        profile = filter_hallucinated_extraction(profile, resume_text)
        profile.summary_claimed_years = extract_claimed_years(resume_text)

        # ── Phase 2: deterministic profile metrics ─────────────────────────
        today = datetime.date.today()
        profile.dates_extracted_ok = dates_present(profile.work_history)
        computed = compute_years_experience(profile.work_history, today)
        profile.years_experience = reconcile_years(computed, profile.summary_claimed_years)

        # ── Phase 3: deterministic matching, relevance, and all sub-scores ─
        # match_skills runs once per requirement-set; relevance uses the
        # already-computed match count instead of recomputing.
        stack = candidate_stack(profile)
        matched_required, missing_required = match_skills(stack, requirements.required_skills)
        matched_nice, _ = match_skills(stack, requirements.nice_to_have_skills)

        profile.years_relevant_experience = compute_relevant_years(
            profile.years_experience, len(matched_required), len(requirements.required_skills),
        )
        skills_score     = compute_skills_match_score(matched_required, requirements.required_skills)
        experience_score = compute_experience_match_score(profile, requirements)
        seniority_score  = compute_seniority_score(profile, requirements)

        # ── Phase 4: LLM narrates the (Python-computed) scores ─────────────
        rationale = self.assess(
            profile=profile,
            requirements=requirements,
            matched_required=matched_required,
            missing_required=missing_required,
            skills_match_score=skills_score,
            experience_match_score=experience_score,
            seniority_match_score=seniority_score,
            dates_extracted_ok=profile.dates_extracted_ok,
        ).out.rationale

        assessment = FitAssessment(
            skills_match_score=skills_score,
            experience_match_score=experience_score,
            seniority_match_score=seniority_score,
            matched_required=matched_required,
            missing_required=missing_required,
            matched_nice_to_have=matched_nice,
            strengths=sorted(set(matched_required) | set(matched_nice)),
            gaps=list(missing_required),
            rationale=rationale,
        )
        assessment.overall_score = combine_subscores(assessment)

        # ── Phase 5: score-calibrated advice (creative_lm for variety) ─────
        with dspy.context(lm=creative_lm):
            suggestions = self.advise(
                assessment=assessment,
                overall_score=assessment.overall_score,
                target_role=target_role,
            ).suggestions

        return dspy.Prediction(
            profile=profile,
            assessment=assessment,
            suggestions=suggestions,
        )
