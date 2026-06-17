# Databricks notebook source
# MAGIC %md
# MAGIC # Elite Athletic AI Advisor: Agent Definition
# MAGIC
# MAGIC This notebook defines the agent instructions, tool functions, guardrails,
# MAGIC and model-serving adapter for the youth basketball advisor.

# COMMAND ----------

# DBTITLE 1,Install MLflow 3.1+ for tracing
# Ensure MLflow 3.1+ is available for GenAI tracing
%pip install -q "mlflow[databricks]>=3.1" --upgrade
dbutils.library.restartPython()

# COMMAND ----------

#restart kernel as recommended for databricks
dbutils.library.restartPython()

# COMMAND ----------

import json
import re
import time
from typing import Any, Dict, List, Optional

import mlflow
from pyspark.sql import functions as F


def _widget(name, default):
    try:
        dbutils.widgets.text(name, default)
        return dbutils.widgets.get(name)
    except Exception:
        return default


CATALOG = _widget("catalog", "main")
SCHEMA = _widget("schema", "default")
MODEL_ENDPOINT = _widget("model_endpoint", "databricks-gpt-oss-120b")
REVIEW_MODEL_ENDPOINT = _widget("review_model_endpoint", "databricks-llama-4-maverick")
ALLOW_MOCK_LLM = _widget("allow_mock_llm", "false").lower() == "true"


def q(table_name):
    """Format a fully qualified table name. DO NOT shadow this with loop variables."""
    return f"`{CATALOG}`.`{SCHEMA}`.`{table_name}`"


def fq_tool_name(tool_name: str) -> str:
    """Format a Unity Catalog tool/function name without SQL quoting."""
    return f"{CATALOG}.{SCHEMA}.{tool_name}"

# Safety: Restore q function if it was shadowed by a loop variable
def _restore_q():
    global q
    if not callable(q):
        def q(table_name):
            return f"`{CATALOG}`.`{SCHEMA}`.`{table_name}`"
    return q


AGENT_SYSTEM_PROMPT = """
You are Elite Athletic AI Advisor, an educational youth basketball development agent.

Mission:
- Create safe, age-appropriate weekly development plans for basketball athletes.
- Treat ages 13-17 as the fully supported youth MVP with parent or coach oversight.
- Treat ages 18+ as a limited adult basketball development preview with self-directed or coach-supported language.
- Use provided tool context before giving final recommendations.
- Explain the reasoning in plain language for athletes, parents, and coaches.

Required safety behavior:
- Do not provide medical advice, diagnosis, rehabilitation protocols, supplement plans, fasting, extreme diets, or weight-cutting guidance.
- If an athlete has any injury note, reduce impact and training volume, recommend guardian/coach oversight, and suggest medical review for pain or recovery concerns.
- Never prescribe more than 5 training days per week for youth athletes.
- Keep adult preview plans conservative and cap planned training at 6 days per week.
- Reject athletes under age 13 for this prototype.
- Include recovery, hydration, sleep, and stretching guidance in every plan.
- Reject requests outside basketball training, recovery, mindset, or general food guidance.

Required output:
- Start with a concise safety note.
- Include a short "Today / This Week" summary.
- Provide a weekly plan by day in a readable table or bullet list.
- Include skill work, strength or conditioning, recovery, and general nutrition guidance.
- Cite the tool context in natural language, such as selected exercise themes, benchmark context, and safety rules.
- End with progress metrics that an athlete, parent, or coach can review next week.
"""

OUT_OF_SCOPE_PATTERNS = [
    r"\bhomework\b",
    r"\bessay\b",
    r"\bexam\b",
    r"\bcrypto\b",
    r"\bstock\b",
    r"\bgambling\b",
    r"\bdating\b",
    r"\bweapon\b",
    r"\bsteroid\b",
    r"\banabolic\b",
    r"\bcreatine\b",
    r"\bsupplement\b",
    r"\bfasting\b",
    r"\bcut weight\b",
    r"\bweight cut\b",
    r"\bmeal replacement\b",
]

IN_SCOPE_TERMS = [
    "basketball",
    "training",
    "workout",
    "plan",
    "recovery",
    "nutrition",
    "food",
    "hydrate",
    "sleep",
    "shooting",
    "defense",
    "agility",
    "speed",
    "conditioning",
    "strength",
    "vertical",
    "jump",
    "ball handling",
    "tryout",
    "practice",
    "mobility",
    "stretch",
]


def rows_to_dicts(df, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if limit is not None:
        df = df.limit(limit)
    return [row.asDict(recursive=True) for row in df.collect()]


def goal_to_tag(goal: str, request: str = "") -> str:
    text = f"{goal or ''} {request or ''}".lower()
    if re.search(r"vertical|jump|explosive|plyometric", text):
        return "vertical_jump"
    if re.search(r"speed|agility|defense|footwork|ball handling|handle", text):
        return "speed_agility"
    if re.search(r"condition|stamina|endurance", text):
        return "conditioning"
    if re.search(r"strength|strong|build muscle", text):
        return "strength"
    if re.search(r"core|balance", text):
        return "core"
    return "general_fitness"


def benchmark_positions(position: str) -> List[str]:
    text = (position or "").lower()
    if "point" in text:
        return ["PG", "G"]
    if "shooting" in text:
        return ["SG", "G"]
    if "small" in text:
        return ["SF", "F", "G-F", "F-G"]
    if "power" in text:
        return ["PF", "F", "F-C", "C-F"]
    if "center" in text:
        return ["C", "C-F", "F-C"]
    return []


@mlflow.trace(name="get_athlete_profile", span_type="TOOL")
def get_athlete_profile(athlete_id: str) -> Dict[str, Any]:
    _restore_q()  # Ensure q is a function, not a shadowed variable
    df = spark.table(q("silver_athlete_profiles")).filter(F.col("athlete_id") == athlete_id)
    rows = rows_to_dicts(df, limit=1)
    if not rows:
        raise ValueError(f"No athlete profile found for athlete_id={athlete_id}")
    return rows[0]


def get_safety_rules() -> List[Dict[str, Any]]:
    # Note: gold_safety_rules table does not exist yet
    # Returning empty list until table is created
    return []


def get_progress_metrics() -> List[Dict[str, Any]]:
    # Note: gold_progress_metric_definitions table does not exist yet
    # Returning empty list until table is created
    return []


def get_feedback_schema() -> List[Dict[str, Any]]:
    # Note: gold_feedback_event_schema table does not exist yet
    # Returning empty list until table is created
    return []


def classify_age_group(age: int) -> str:
    if 13 <= age <= 17:
        return "youth"
    if age >= 18:
        return "adult"
    return "unsupported"


@mlflow.trace(name="check_safety", span_type="TOOL")
def check_safety(profile: Dict[str, Any], request: str) -> Dict[str, Any]:
    request_text = (request or "").lower()
    sport = (profile.get("sport") or "").lower()
    injury_status = (profile.get("injury_status") or "none").lower()
    available_days = int(profile.get("available_days") or 0)
    age = int(profile.get("age") or 0)
    age_group = profile.get("age_group") or classify_age_group(age)

    for pattern in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, request_text):
            return {
                "allowed": False,
                "reason": "The request is outside the safe scope for a basketball development advisor.",
                "response": (
                    "I can help with safe basketball training, recovery, and general food guidance, "
                    "but I cannot help with that request. Please ask for a basketball development plan, "
                    "recovery checklist, or age-appropriate nutrition guidance instead."
                ),
            }

    if request_text and not any(term in request_text for term in IN_SCOPE_TERMS):
        return {
            "allowed": False,
            "reason": "The request is unrelated to youth basketball development.",
            "response": (
                "I am scoped to youth basketball development. I can help with training, recovery, "
                "skill work, conditioning, or general nutrition guidance for a basketball athlete."
            ),
        }

    if sport != "basketball":
        return {
            "allowed": False,
            "reason": f"Unsupported sport: {sport}",
            "response": "This prototype currently supports basketball only.",
        }

    if age < 13:
        return {
            "allowed": False,
            "reason": f"Unsupported age: {age}",
            "response": (
                "This prototype supports youth basketball athletes ages 13-17 with parent or coach oversight "
                "and a limited adult preview for ages 18+. For athletes under 13, please work directly with a "
                "qualified coach, parent, or clinician."
            ),
        }

    if age_group == "adult":
        oversight_model = "Self-directed or coach-supported review is recommended."
        max_training_days = 6
    else:
        oversight_model = "Parent or coach oversight is required."
        max_training_days = 5

    constraints = [
        oversight_model,
        "Do not include medical advice, supplements, fasting, weight cutting, or extreme diets.",
        "Include recovery, hydration, sleep, and stretching guidance.",
    ]

    safe_training_days = min(max(available_days, 1), max_training_days)
    if available_days > max_training_days:
        constraints.append(
            f"Cap planned training at {max_training_days} days per week even if the profile lists more availability."
        )

    if age_group == "adult":
        constraints.append("Label this as an adult preview because the core MVP remains youth-focused.")

    if injury_status != "none":
        constraints.append(
            "Injury status is present, so avoid high-impact work, reduce volume, and recommend coach or medical review."
        )

    return {
        "allowed": True,
        "reason": "Request is in scope.",
        "age_group": age_group,
        "oversight_model": oversight_model,
        "safe_training_days": safe_training_days,
        "constraints": constraints,
    }


@mlflow.trace(name="retrieve_exercises", span_type="RETRIEVER")
def retrieve_exercises(goal: str, equipment: str, injury_status: str, limit: int = 6) -> List[Dict[str, Any]]:
    _restore_q()  # Ensure q is a function
    goal_tag = goal_to_tag(goal)
    equipment_text = (equipment or "").lower()
    injury_text = (injury_status or "none").lower()

    df = spark.table(q("gold_exercise_recommendations"))
    df = df.filter((F.col("goal_tag") == goal_tag) | (F.col("goal_tag") == "general_fitness"))

    if "gym" not in equipment_text:
        df = df.filter(~F.col("equipment").rlike("barbell|machine|cable|kettlebell"))
    if "resistance band" not in equipment_text and "bands" not in equipment_text:
        df = df.filter(~F.col("equipment").rlike("band"))
    if injury_text != "none":
        df = df.filter(~F.lower(F.col("agent_text")).rlike("jump|plyometric|sprint|max effort|explosive"))

    return rows_to_dicts(df.orderBy(F.rand(seed=7)), limit=limit)


@mlflow.trace(name="recommend_exercises", span_type="RETRIEVER")
def recommend_exercises(
    target_goal: str,
    target_body_part: str = "",
    equipment: str = "",
    injury_status: str = "none",
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Python adapter for the UC SQL function main.default.recommend_exercises."""
    _restore_q()
    goal_text = (target_goal or "").lower()
    body_part_text = (target_body_part or "").lower()
    equipment_text = (equipment or "").lower()
    injury_text = (injury_status or "none").lower()

    df = spark.table(q("gold_exercise_recommendations"))
    if goal_text or body_part_text:
        conditions = []
        if body_part_text:
            conditions.append(F.lower(F.col("body_part")).contains(body_part_text))
        if goal_text:
            conditions.extend([
                F.lower(F.col("goal_tag")).contains(goal_text),
                F.lower(F.col("agent_text")).contains(goal_text),
            ])
        combined_condition = conditions[0]
        for condition in conditions[1:]:
            combined_condition = combined_condition | condition
        df = df.filter(combined_condition)
    if "gym" not in equipment_text:
        df = df.filter(~F.col("equipment").rlike("barbell|machine|cable|kettlebell"))
    if "resistance band" not in equipment_text and "bands" not in equipment_text:
        df = df.filter(~F.col("equipment").rlike("band"))
    if injury_text != "none":
        df = df.filter(~F.lower(F.col("agent_text")).rlike("jump|plyometric|sprint|max effort|explosive"))

    return rows_to_dicts(df.limit(limit))


@mlflow.trace(name="get_basketball_benchmark", span_type="RETRIEVER")
def get_basketball_benchmark(position: str) -> List[Dict[str, Any]]:
    _restore_q()  # Ensure q is a function
    candidates = benchmark_positions(position)
    df = spark.table(q("gold_basketball_benchmarks"))
    if candidates:
        filtered = df.filter(F.col("position").isin(candidates))
        rows = rows_to_dicts(filtered, limit=5)
        if rows:
            return rows
    # Return all benchmarks ordered by points (no sample_size column available)
    return rows_to_dicts(df.orderBy(F.desc("avg_points")), limit=5)


@mlflow.trace(name="lookup_nutrition", span_type="RETRIEVER")
def lookup_nutrition(goal: str, injury_status: str) -> List[Dict[str, Any]]:
    _restore_q()  # Ensure q is a function
    # Query the UC function lookup_nutrition_guidance
    goal_tag = goal_to_tag(goal)
    injury_text = (injury_status or "none").lower()
    
    df = spark.sql(f"""
        SELECT * 
        FROM {fq_tool_name('lookup_nutrition_guidance')}('{goal_tag}', '{injury_text}')
    """)
    return rows_to_dicts(df, limit=10)


@mlflow.trace(name="lookup_nutrient", span_type="RETRIEVER")
def lookup_nutrient(nutrient_query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Python adapter for the UC SQL function main.default.lookup_nutrient."""
    _restore_q()
    return rows_to_dicts(
        spark.table(q("gold_nutrition_lookup"))
        .filter(F.lower(F.col("nutrient_name")).contains((nutrient_query or "").lower()))
        .select("food_nutrient_id", "nutrient_name", "adjusted_amount", "lab_method_id")
        .limit(limit)
    )


UC_TOOL_NAMES = [
    fq_tool_name("recommend_exercises"),
    fq_tool_name("lookup_nutrient"),
    fq_tool_name("lookup_nutrition_guidance"),
    fq_tool_name("get_basketball_benchmark"),
]


@mlflow.trace(name="build_tool_context", span_type="TOOL")
def build_tool_context(profile: Dict[str, Any], request: str) -> Dict[str, Any]:
    safety = check_safety(profile, request)
    if not safety["allowed"]:
        return {"safety": safety}

    return {
        "profile": profile,
        "safety": safety,
        "tool_names_used": UC_TOOL_NAMES,
        "safety_rules": get_safety_rules(),
        "progress_metrics": get_progress_metrics(),
        "feedback_schema": get_feedback_schema(),
        "recommended_exercises": recommend_exercises(
            target_goal=goal_to_tag(profile.get("goal", ""), request),
            target_body_part="",
            equipment=profile.get("equipment", ""),
            injury_status=profile.get("injury_status", "none"),
            limit=6,
        ),
        "basketball_benchmarks": get_basketball_benchmark(profile.get("position", "")),
        "nutrition_guidance": lookup_nutrition(
            goal=profile.get("goal", ""),
            injury_status=profile.get("injury_status", "none"),
        ),
        "nutrition_lookup_records": lookup_nutrient("protein", limit=5),
    }


TOOL_REGISTRY = {
    fq_tool_name("recommend_exercises"): recommend_exercises,
    fq_tool_name("lookup_nutrient"): lookup_nutrient,
    fq_tool_name("get_basketball_benchmark"): get_basketball_benchmark,
    "recommend_exercises": recommend_exercises,
    "lookup_nutrient": lookup_nutrient,
    "get_athlete_profile": get_athlete_profile,
    "check_safety": check_safety,
    "retrieve_exercises": retrieve_exercises,
    "get_basketball_benchmark": get_basketball_benchmark,
    "lookup_nutrition": lookup_nutrition,
    "get_progress_metrics": get_progress_metrics,
    "get_feedback_schema": get_feedback_schema,
}


def load_gold_athlete_profiles(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load profile rows for natural-language intake matching."""
    _restore_q()
    df = spark.table(q("silver_athlete_profiles"))
    return rows_to_dicts(df, limit=limit)

# COMMAND ----------

# DBTITLE 1,Comprehensive User Profile Schema and Enhanced Prompts
# ============================================================================
# COMPREHENSIVE USER PROFILE INTAKE SYSTEM
# ============================================================================

# Enhanced User Profile Schema
USER_PROFILE_SCHEMA = {
    # Core Demographics (Required)
    "age": "int (required)",
    "sport": "string (required, must be 'basketball')",
    "primary_goal": "string (required: e.g. improve agility, increase vertical, build strength)",
    "training_days_available": "int (required: days per week)",
    
    # Physical Attributes (Optional but valuable)
    "position": "string (optional: Point Guard, Shooting Guard, Small Forward, Power Forward, Center)",
    "height": "string (optional: e.g. '6-2' or '188cm')",
    "weight": "string (optional: e.g. '175 lbs' or '79 kg')",
    "build": "string (optional: lean, average, muscular, stocky)",
    "dominant_hand": "string (optional: left, right, both)",
    
    # Experience & Skill Level (Optional but important)
    "years_playing": "int (optional: how many years playing basketball)",
    "competition_level": "string (optional: recreational, middle school, high school JV, high school varsity, travel/AAU, club)",
    "skill_level": "string (optional: beginner, intermediate, advanced)",
    "current_fitness_level": "string (optional: beginner, moderate, athletic)",
    
    # Training Goals & Context (Optional)
    "secondary_goal": "string (optional)",
    "specific_weaknesses": "string (optional: what they struggle with most)",
    "season_phase": "string (optional: off-season, pre-season, in-season)",
    "upcoming_events": "string (optional: e.g. tryouts in 3 weeks, tournament next month)",
    
    # Health & Recovery (Optional but critical)
    "injury_status": "string (optional: none, or brief injury description)",
    "previous_injuries": "string (optional: relevant injury history)",
    "sleep_hours_avg": "int (optional: average hours per night, critical for youth)",
    
    # Equipment & Nutrition (Optional)
    "equipment_access": "string (optional: e.g. home gym, dumbbells, basketball court)",
    "nutrition_goal": "string (optional: e.g. general healthy eating, gain muscle, maintain weight)",
    "dietary_constraints": "string (optional: e.g. vegetarian, food allergies)",
    
    # Oversight (Optional)
    "coach_or_parent_oversight": "string (optional: yes/no or description)",
    "free_text_notes": "string (optional: any additional context)",
}

INTAKE_EXTRACTION_PROMPT = """
You are a profile extraction assistant for a youth basketball development advisor.

Your task: Convert natural language user input into a comprehensive structured JSON profile.

**REQUIRED fields (must extract or mark as unknown):**
- age (integer)
- sport (must be "basketball")
- primary_goal (what the athlete wants to improve)
- training_days_available (how many days per week they can train)

**OPTIONAL fields (extract if mentioned, otherwise use "unknown"):**

Physical:
- position, height, weight, build, dominant_hand

Experience:
- years_playing (integer or unknown)
- competition_level (recreational, middle school, high school JV/varsity, travel/AAU, club)
- skill_level (beginner, intermediate, advanced)
- current_fitness_level (beginner, moderate, athletic)

Context:
- secondary_goal, specific_weaknesses, season_phase, upcoming_events

Health:
- injury_status (use "none" if no injury mentioned), previous_injuries
- sleep_hours_avg (integer or unknown)

Equipment:
- equipment_access, nutrition_goal, dietary_constraints

Oversight:
- coach_or_parent_oversight

**Extraction Rules:**
1. Extract information directly from the user's message
2. Do NOT invent information that isn't mentioned
3. Use "unknown" for optional fields that aren't mentioned (NOT null or empty)
4. For injury_status, use "none" if no injury is mentioned
5. Be conservative: if unclear whether something is an injury, mark it as potential injury
6. For years_playing, if they say "played for X years" extract X as integer
7. For competition_level, infer from context (e.g., "high school team" → "high school varsity")
8. Store original user input in free_text_notes (first 200 chars)
9. Return ONLY valid JSON, no explanation text

**Example input:**
"I'm 16, been playing guard for 5 years on my high school team. Want to improve my explosiveness and ball handling for varsity tryouts next month. Have some knee soreness. Can train 4 days a week. Sleep about 7 hours. Have dumbbells at home."

**Example output:**
{
  "age": 16,
  "sport": "basketball",
  "position": "Guard",
  "height": "unknown",
  "weight": "unknown",
  "build": "unknown",
  "dominant_hand": "unknown",
  "years_playing": 5,
  "competition_level": "high school varsity",
  "skill_level": "unknown",
  "current_fitness_level": "unknown",
  "primary_goal": "improve explosiveness",
  "secondary_goal": "improve ball handling",
  "specific_weaknesses": "unknown",
  "season_phase": "pre-season",
  "upcoming_events": "varsity tryouts next month",
  "injury_status": "knee soreness",
  "previous_injuries": "unknown",
  "sleep_hours_avg": 7,
  "training_days_available": 4,
  "equipment_access": "dumbbells at home",
  "nutrition_goal": "unknown",
  "dietary_constraints": "unknown",
  "coach_or_parent_oversight": "unknown",
  "free_text_notes": "I'm 16, been playing guard for 5 years on my high school team. Want to improve my explosiveness and ball handling for varsity tryouts next month. Have some knee soreness. Can train 4 days a week..."
}

Now extract the profile from the user's input below.
"""

PROFILE_MATCHING_PROMPT = """
You are analyzing athlete profile matches for personalized coaching.

You will receive:
1. A new user profile (with comprehensive details)
2. Top 3 matching athlete profiles from the database with match scores
3. Deterministic scoring breakdown

Your task: Explain in 2-3 sentences why these matches were selected and what athlete archetype this represents.

Consider in your explanation:
- Age group and experience level
- Position and playing style
- Primary development focus
- Competition level and training context
- Season phase and upcoming events (if relevant)
- Any injury considerations that influenced matching
- Fitness level and training capacity

Examples of comprehensive archetypes:
- "Experienced youth guard: 3+ years playing, high school level, agility and ball-handling development"
- "Beginner youth forward: first-year player, recreational level, fundamental strength and mobility"
- "Advanced youth post: varsity level, pre-season preparation, vertical jump and strength focus"
- "Adult recreational athlete: conditioning maintenance, pickup game performance"

Be specific and use the full profile context to create a rich, personalized archetype description.
"""

REJECTION_PROMPT = """
You are rejecting a request that falls outside the safe scope of a youth basketball development advisor.

Provide a brief, polite rejection that:
1. Acknowledges what the user asked for
2. Explains why it's outside scope (age, sport, medical, supplements, academic)
3. Redirects to what you CAN help with

Be warm but firm. Keep it under 100 words.
"""

PLAN_REVIEW_PROMPT = """
You are the safety and quality reviewer for Elite Athletic AI Advisor.

Review the drafted plan using the athlete profile and tool context. Return ONLY valid JSON with:
- passes: boolean
- safety_score: integer from 1 to 5
- grounding_score: integer from 1 to 5
- clarity_score: integer from 1 to 5
- requires_human_review: boolean
- reviewer_notes: short string
- revision_requests: array of short strings

The plan fails if it recommends medical treatment, diagnosis, supplements, fasting,
weight cutting, extreme diets, unsafe high-impact work for an injury note, too many
training days for the athlete age group, or content outside basketball development.
"""

# COMMAND ----------

# DBTITLE 1,Enhanced Profile Extraction Function
@mlflow.trace(name="extract_user_profile", span_type="PARSER")
def extract_user_profile(user_input: str, model_endpoint: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract comprehensive structured profile from natural language user input using LLM.
    """
    endpoint = model_endpoint or MODEL_ENDPOINT
    
    # Mock extraction for demo/testing (enhanced heuristics)
    if ALLOW_MOCK_LLM or not endpoint or endpoint.startswith("TODO_"):
        profile = {
            # Core required
            "age": None,
            "sport": "basketball",
            "primary_goal": "general basketball development",
            "training_days_available": 3,
            
            # Physical
            "position": "unknown",
            "height": "unknown",
            "weight": "unknown",
            "build": "unknown",
            "dominant_hand": "unknown",
            
            # Experience
            "years_playing": "unknown",
            "competition_level": "unknown",
            "skill_level": "unknown",
            "current_fitness_level": "unknown",
            
            # Context
            "secondary_goal": "unknown",
            "specific_weaknesses": "unknown",
            "season_phase": "unknown",
            "upcoming_events": "unknown",
            
            # Health
            "injury_status": "none",
            "previous_injuries": "unknown",
            "sleep_hours_avg": "unknown",
            
            # Equipment & Nutrition
            "equipment_access": "unknown",
            "nutrition_goal": "unknown",
            "dietary_constraints": "unknown",
            
            # Oversight
            "coach_or_parent_oversight": "unknown",
            "free_text_notes": user_input[:200],
        }
        
        user_lower = user_input.lower()
        
        # Extract age
        age_match = re.search(r"\b(\d{1,2})\s*(?:years?\s*old|yr|yo)?\b", user_lower)
        if age_match:
            profile["age"] = int(age_match.group(1))
        
        # Extract training days
        days_match = re.search(r"(\d)\s*days?\s*(?:per\s*week|a\s*week|weekly)?", user_lower)
        if days_match:
            profile["training_days_available"] = int(days_match.group(1))
        
        # Extract position
        positions = ["point guard", "shooting guard", "small forward", "power forward", "center", "guard", "forward"]
        for pos in positions:
            if pos in user_lower:
                profile["position"] = pos.title()
                break
        
        # Extract years playing
        years_match = re.search(r"(?:played|playing|been playing)(?:\s+\w+){0,3}\s+(\d+)\s*years?", user_lower)
        if years_match:
            profile["years_playing"] = int(years_match.group(1))
        
        # Extract competition level
        if "varsity" in user_lower or "var" in user_lower:
            profile["competition_level"] = "high school varsity"
        elif "jv" in user_lower or "junior varsity" in user_lower:
            profile["competition_level"] = "high school JV"
        elif "high school" in user_lower or "hs team" in user_lower:
            profile["competition_level"] = "high school"
        elif "middle school" in user_lower or "ms team" in user_lower:
            profile["competition_level"] = "middle school"
        elif "travel" in user_lower or "aau" in user_lower:
            profile["competition_level"] = "travel/AAU"
        elif "club" in user_lower:
            profile["competition_level"] = "club"
        elif "recreational" in user_lower or "pickup" in user_lower:
            profile["competition_level"] = "recreational"
        
        # Extract season phase
        if "off-season" in user_lower or "offseason" in user_lower:
            profile["season_phase"] = "off-season"
        elif "pre-season" in user_lower or "preseason" in user_lower or "before season" in user_lower:
            profile["season_phase"] = "pre-season"
        elif "in-season" in user_lower or "in season" in user_lower or "during season" in user_lower:
            profile["season_phase"] = "in-season"
        
        # Extract upcoming events
        events_match = re.search(r"(tryouts?|tournament|game|championship|season)(?:\s+\w+){0,5}", user_lower)
        if events_match:
            profile["upcoming_events"] = events_match.group(0)
        
        # Extract sleep hours
        sleep_match = re.search(r"(\d+)\s*(?:hours?)?\s*(?:of)?\s*sleep", user_lower)
        if sleep_match:
            profile["sleep_hours_avg"] = int(sleep_match.group(1))
        
        # Extract dominant hand
        if "left hand" in user_lower or "lefty" in user_lower:
            profile["dominant_hand"] = "left"
        elif "right hand" in user_lower or "righty" in user_lower:
            profile["dominant_hand"] = "right"
        elif "both hands" in user_lower or "ambidextrous" in user_lower:
            profile["dominant_hand"] = "both"
        
        # Extract build
        if "lean" in user_lower or "skinny" in user_lower or "thin" in user_lower:
            profile["build"] = "lean"
        elif "muscular" in user_lower or "strong" in user_lower:
            profile["build"] = "muscular"
        elif "stocky" in user_lower or "thick" in user_lower:
            profile["build"] = "stocky"
        
        # Extract fitness level
        if "beginner" in user_lower or "new to" in user_lower or "just started" in user_lower:
            profile["current_fitness_level"] = "beginner"
        elif "athletic" in user_lower or "fit" in user_lower or "good shape" in user_lower:
            profile["current_fitness_level"] = "athletic"
        elif "moderate" in user_lower or "average" in user_lower:
            profile["current_fitness_level"] = "moderate"
        
        # Extract injury
        injury_keywords = ["injury", "pain", "sore", "hurt", "sprain", "strain", "recovering", "recovery"]
        for keyword in injury_keywords:
            if keyword in user_lower:
                injury_match = re.search(rf"\b\w*\s*{keyword}\w*\b", user_lower)
                if injury_match:
                    profile["injury_status"] = injury_match.group(0)
                else:
                    profile["injury_status"] = keyword
                break
        
        # Extract goals
        goal_keywords = {
            "agility": "improve agility",
            "speed": "improve speed",
            "vertical": "increase vertical jump",
            "jump": "improve jumping",
            "explosiveness": "improve explosiveness",
            "quickness": "improve quickness",
            "strength": "build strength",
            "conditioning": "improve conditioning",
            "endurance": "improve endurance",
            "ball handling": "improve ball handling",
            "dribbling": "improve dribbling",
            "shooting": "improve shooting",
            "defense": "improve defense",
        }
        found_goals = []
        for keyword, goal in goal_keywords.items():
            if keyword in user_lower:
                found_goals.append(goal)
        if found_goals:
            profile["primary_goal"] = found_goals[0]
            if len(found_goals) > 1:
                profile["secondary_goal"] = found_goals[1]
        
        return profile
    
    # LLM-based extraction
    import mlflow.deployments
    
    messages = [
        {"role": "system", "content": INTAKE_EXTRACTION_PROMPT},
        {"role": "user", "content": user_input},
    ]
    
    client = mlflow.deployments.get_deploy_client("databricks")
    response = client.predict(
        endpoint=endpoint,
        inputs={
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 800,
        },
    )
    
    response_text = extract_chat_content(response)
    
    # Parse JSON from response
    try:
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*({.*?})\s*```", response_text, re.DOTALL)
        if json_match:
            profile = json.loads(json_match.group(1))
        else:
            # Try direct JSON parse
            profile = json.loads(response_text)
        return profile
    except json.JSONDecodeError:
        # Fallback to heuristic extraction
        return extract_user_profile(user_input, model_endpoint="TODO_HEURISTIC_EXTRACTION")

# COMMAND ----------

# DBTITLE 1,Enhanced Profile Validation Function
@mlflow.trace(name="validate_user_profile", span_type="TOOL")
def validate_user_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate comprehensive user profile for safety, eligibility, and completeness.
    Returns validation result with allowed status, reason, missing fields, and clarifying questions.
    """
    validation = {
        "allowed": True,
        "reason": "Profile is valid",
        "missing_fields": [],
        "clarifying_questions": [],
        "warnings": [],
        "recommendations": [],
    }
    
    # Check required fields
    age = profile.get("age")
    sport = (profile.get("sport") or "").lower()
    primary_goal = profile.get("primary_goal") or ""
    training_days = profile.get("training_days_available")
    
    # Validate age
    if not age or age == "unknown":
        validation["missing_fields"].append("age")
        validation["clarifying_questions"].append("How old are you?")
        validation["allowed"] = False
        validation["reason"] = "Missing required field: age"
        return validation
    
    try:
        age = int(age)
    except (ValueError, TypeError):
        validation["allowed"] = False
        validation["reason"] = "Invalid age value"
        return validation
    
    if age < 13:
        validation["allowed"] = False
        validation["reason"] = "Age below minimum (13)"
        validation["response"] = (
            "This prototype supports youth basketball athletes ages 13-17 with parent or coach oversight "
            "and a limited adult preview for ages 18+. For athletes under 13, please work directly with a "
            "qualified coach, parent, or clinician."
        )
        return validation
    
    # Classify age group
    age_group = classify_age_group(age)
    if age_group == "unsupported":
        validation["allowed"] = False
        validation["reason"] = "Unsupported age"
        return validation
    
    validation["age_group"] = age_group
    
    # Validate sport
    if not sport or sport == "unknown":
        validation["missing_fields"].append("sport")
        validation["clarifying_questions"].append("What sport do you play?")
        validation["allowed"] = False
        validation["reason"] = "Missing required field: sport"
        return validation
    
    if sport != "basketball":
        validation["allowed"] = False
        validation["reason"] = f"Unsupported sport: {sport}"
        validation["response"] = "This prototype currently supports basketball only."
        return validation
    
    # Validate primary goal
    if not primary_goal or primary_goal == "unknown":
        validation["missing_fields"].append("primary_goal")
        validation["clarifying_questions"].append(
            "What is your main goal? (e.g., improve agility, increase vertical jump, build strength)"
        )
        validation["allowed"] = False
        validation["reason"] = "Missing required field: primary_goal"
        return validation
    
    # Check for out-of-scope goals in extracted goal AND free text notes (original user input)
    goal_text = f"{primary_goal} {profile.get('secondary_goal', '')}".lower()
    free_text = (profile.get('free_text_notes') or "").lower()
    full_text = f"{goal_text} {free_text}"
    
    for pattern in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, full_text):
            validation["allowed"] = False
            validation["reason"] = "Request contains out-of-scope topics (medical/supplements/academic)"
            validation["response"] = (
                "I can help with safe basketball training, recovery, and general food guidance, "
                "but I cannot help with that request. Please ask for a basketball development plan, "
                "recovery checklist, or age-appropriate nutrition guidance instead."
            )
            return validation
    
    # Validate training days
    if not training_days or training_days == "unknown":
        validation["missing_fields"].append("training_days_available")
        validation["clarifying_questions"].append("How many days per week can you train?")
        validation["allowed"] = False
        validation["reason"] = "Missing required field: training_days_available"
        return validation
    
    try:
        training_days = int(training_days)
        if training_days < 1:
            validation["warnings"].append("Training days is less than 1, using minimum of 1 day")
            training_days = 1
        elif training_days > 7:
            validation["warnings"].append("Training days exceeds 7, capping at safe limits")
    except (ValueError, TypeError):
        validation["allowed"] = False
        validation["reason"] = "Invalid training_days_available value"
        return validation
    
    # Check injury status
    injury_status = (profile.get("injury_status") or "none").lower()
    if injury_status != "none" and injury_status != "unknown":
        validation["warnings"].append(
            f"Injury noted: {injury_status}. Plan will avoid high-impact work and recommend medical review."
        )
        validation["injury_present"] = True
    else:
        validation["injury_present"] = False
    
    # Check sleep hours (critical for youth)
    sleep_hours = profile.get("sleep_hours_avg")
    if sleep_hours and sleep_hours != "unknown":
        try:
            sleep_hours = int(sleep_hours)
            if age_group == "youth" and sleep_hours < 7:
                validation["warnings"].append(
                    f"Sleep ({sleep_hours}h/night) is below recommended 8-10h for youth athletes. "
                    "Plan will reduce training volume and emphasize recovery."
                )
                validation["low_sleep"] = True
            elif sleep_hours < 6:
                validation["warnings"].append(
                    f"Sleep ({sleep_hours}h/night) is critically low. Plan will be conservative with recovery focus."
                )
                validation["low_sleep"] = True
            else:
                validation["low_sleep"] = False
        except (ValueError, TypeError):
            pass
    
    # Check experience level and competition
    years_playing = profile.get("years_playing")
    competition_level = profile.get("competition_level") or "unknown"
    
    if years_playing and years_playing != "unknown":
        try:
            years_playing = int(years_playing)
            if years_playing == 0 or years_playing == 1:
                validation["recommendations"].append("Beginner detected - plan will focus on fundamentals and safe progression")
            elif years_playing >= 5:
                validation["recommendations"].append("Experienced player - plan can include advanced progressions")
        except (ValueError, TypeError):
            pass
    
    if "varsity" in competition_level.lower() or "aau" in competition_level.lower() or "travel" in competition_level.lower():
        validation["recommendations"].append("Competitive level detected - plan will address high-performance needs")
    
    # Check season phase
    season_phase = (profile.get("season_phase") or "unknown").lower()
    if season_phase == "in-season":
        validation["recommendations"].append("In-season: Plan will focus on maintenance and injury prevention")
    elif season_phase == "off-season":
        validation["recommendations"].append("Off-season: Plan can include strength building and skill development")
    elif season_phase == "pre-season":
        validation["recommendations"].append("Pre-season: Plan will ramp up conditioning and game preparation")
    
    # Check upcoming events
    upcoming_events = (profile.get("upcoming_events") or "unknown").lower()
    if "tryout" in upcoming_events:
        validation["recommendations"].append("Tryouts coming up - plan will prioritize skill sharpness and confidence")
    
    # Set oversight requirements
    if age_group == "youth":
        validation["oversight_required"] = "Parent or coach oversight is required"
        validation["max_training_days"] = 5
    else:
        validation["oversight_required"] = "Self-directed or coach-supported review is recommended"
        validation["max_training_days"] = 6
    
    # Optional field warnings (now more nuanced)
    if profile.get("position") == "unknown" or not profile.get("position"):
        validation["warnings"].append("Position not specified - will use general basketball recommendations")
    
    if profile.get("equipment_access") == "unknown" or not profile.get("equipment_access"):
        validation["warnings"].append("Equipment access not specified - will assume bodyweight exercises")
    
    if profile.get("current_fitness_level") == "unknown" or not profile.get("current_fitness_level"):
        validation["recommendations"].append("Fitness level unknown - plan will start conservatively")
    
    return validation

# COMMAND ----------

# DBTITLE 1,Enhanced Profile Matching Function
@mlflow.trace(name="match_user_profile", span_type="RETRIEVER")
def match_user_profile(
    extracted_profile: Dict[str, Any],
    gold_profiles: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Match comprehensive user profile against gold athlete profiles using weighted scoring.
    
    Scoring dimensions (enhanced):
    - Age similarity (20 points)
    - Position match (15 points)
    - Goal alignment (25 points)
    - Injury status match (15 points)
    - Equipment compatibility (10 points)
    - Training days alignment (10 points)
    - Experience level match (5 points)
    
    Total: 100 points possible
    """
    if gold_profiles is None:
        gold_profiles = load_gold_athlete_profiles()
    
    user_age = extracted_profile.get("age")
    user_position = (extracted_profile.get("position") or "").lower()
    user_goal = (extracted_profile.get("primary_goal") or "").lower()
    user_injury = (extracted_profile.get("injury_status") or "none").lower()
    user_equipment = (extracted_profile.get("equipment_access") or "").lower()
    user_training_days = extracted_profile.get("training_days_available") or 3
    user_years_playing = extracted_profile.get("years_playing")
    user_competition = (extracted_profile.get("competition_level") or "").lower()
    user_skill_level = (extracted_profile.get("skill_level") or "").lower()
    
    matches = []
    
    for athlete in gold_profiles:
        score = 0
        score_breakdown = {}
        
        # Age similarity (20 points)
        if user_age:
            athlete_age = athlete.get("age")
            if athlete_age:
                age_diff = abs(int(user_age) - int(athlete_age))
                if age_diff == 0:
                    age_score = 20
                elif age_diff == 1:
                    age_score = 18
                elif age_diff == 2:
                    age_score = 15
                elif age_diff <= 3:
                    age_score = 10
                else:
                    age_score = 5
                score += age_score
                score_breakdown["age"] = age_score
        
        # Position match (15 points)
        athlete_position = (athlete.get("position") or "").lower()
        if user_position != "unknown" and athlete_position:
            if user_position == athlete_position:
                score += 15
                score_breakdown["position"] = 15
            elif "guard" in user_position and "guard" in athlete_position:
                score += 12
                score_breakdown["position"] = 12
            elif "forward" in user_position and "forward" in athlete_position:
                score += 12
                score_breakdown["position"] = 12
            else:
                score += 5
                score_breakdown["position"] = 5
        
        # Goal alignment (25 points)
        athlete_goal = (athlete.get("goal") or "").lower()
        if athlete_goal:
            # Check for keyword overlap
            user_goal_words = set(user_goal.split())
            athlete_goal_words = set(athlete_goal.split())
            overlap = user_goal_words & athlete_goal_words
            
            if len(overlap) >= 2:
                score += 25
                score_breakdown["goal"] = 25
            elif len(overlap) == 1:
                score += 18
                score_breakdown["goal"] = 18
            else:
                # Semantic similarity (basic)
                agility_keywords = {"agility", "speed", "quickness", "explosiveness"}
                strength_keywords = {"strength", "power", "muscle"}
                skill_keywords = {"ball", "handling", "dribbling", "shooting"}
                
                user_has_agility = bool(user_goal_words & agility_keywords)
                athlete_has_agility = bool(athlete_goal_words & agility_keywords)
                user_has_strength = bool(user_goal_words & strength_keywords)
                athlete_has_strength = bool(athlete_goal_words & strength_keywords)
                user_has_skill = bool(user_goal_words & skill_keywords)
                athlete_has_skill = bool(athlete_goal_words & skill_keywords)
                
                if (user_has_agility and athlete_has_agility) or \
                   (user_has_strength and athlete_has_strength) or \
                   (user_has_skill and athlete_has_skill):
                    score += 15
                    score_breakdown["goal"] = 15
                else:
                    score += 5
                    score_breakdown["goal"] = 5
        
        # Injury status match (15 points)
        athlete_injury = (athlete.get("injury_status") or "none").lower()
        if user_injury == "none" and athlete_injury == "none":
            score += 15
            score_breakdown["injury"] = 15
        elif user_injury != "none" and athlete_injury != "none":
            score += 12
            score_breakdown["injury"] = 12
        else:
            score += 5
            score_breakdown["injury"] = 5
        
        # Equipment compatibility (10 points)
        athlete_equipment = (athlete.get("equipment") or "").lower()
        if athlete_equipment:
            equipment_keywords = athlete_equipment.split()
            matching_equipment = sum(1 for keyword in equipment_keywords if keyword in user_equipment)
            if matching_equipment >= 2:
                score += 10
                score_breakdown["equipment"] = 10
            elif matching_equipment == 1:
                score += 7
                score_breakdown["equipment"] = 7
            else:
                score += 3
                score_breakdown["equipment"] = 3
        
        # Training days alignment (10 points)
        athlete_training_days = athlete.get("available_days") or athlete.get("training_days_available") or 3
        try:
            days_diff = abs(int(user_training_days) - int(athlete_training_days))
            if days_diff == 0:
                score += 10
                score_breakdown["training_days"] = 10
            elif days_diff == 1:
                score += 7
                score_breakdown["training_days"] = 7
            else:
                score += 3
                score_breakdown["training_days"] = 3
        except (ValueError, TypeError):
            score += 3
            score_breakdown["training_days"] = 3
        
        # Experience level match (5 points) - NEW
        athlete_skill = (athlete.get("experience_level") or athlete.get("skill_level") or "").lower()
        if user_years_playing and user_years_playing != "unknown":
            try:
                years = int(user_years_playing)
                # Map years to skill level
                if years <= 1:
                    inferred_skill = "beginner"
                elif years <= 3:
                    inferred_skill = "intermediate"
                else:
                    inferred_skill = "advanced"
                
                if athlete_skill == inferred_skill:
                    score += 5
                    score_breakdown["experience"] = 5
                else:
                    score += 2
                    score_breakdown["experience"] = 2
            except (ValueError, TypeError):
                score += 2
                score_breakdown["experience"] = 2
        elif user_skill_level != "unknown" and athlete_skill:
            if user_skill_level == athlete_skill:
                score += 5
                score_breakdown["experience"] = 5
            else:
                score += 2
                score_breakdown["experience"] = 2
        else:
            score += 2
            score_breakdown["experience"] = 2
        
        matches.append({
            "athlete_id": athlete.get("athlete_id"),
            "athlete_profile": athlete,
            "match_score": score,
            "score_breakdown": score_breakdown,
        })
    
    # Sort by score (descending)
    matches.sort(key=lambda x: x["match_score"], reverse=True)
    
    # Generate comprehensive archetype description
    top_match = matches[0] if matches else None
    archetype = "General basketball athlete"
    
    if top_match:
        age_group = "Youth" if user_age and int(user_age) < 18 else "Adult"
        position_str = user_position.title() if user_position != "unknown" else "player"
        
        # Experience context
        experience_str = ""
        if user_years_playing and user_years_playing != "unknown":
            try:
                years = int(user_years_playing)
                if years <= 1:
                    experience_str = "beginner, first-year"
                elif years <= 3:
                    experience_str = f"{years}-year, developing"
                else:
                    experience_str = f"{years}+ year, experienced"
            except (ValueError, TypeError):
                pass
        
        # Competition context
        competition_str = ""
        if user_competition != "unknown":
            competition_str = user_competition
        
        # Goal focus
        goal_focus = user_goal.replace("improve ", "").replace("increase ", "").replace("build ", "")
        
        # Build archetype
        parts = [f"{age_group} {position_str}"]
        if experience_str:
            parts.append(experience_str)
        if competition_str:
            parts.append(competition_str + " level")
        parts.append(goal_focus + " development")
        
        archetype = ": ".join(parts)
    
    return {
        "top_matches": matches[:3],
        "all_matches": matches,
        "archetype": archetype,
        "best_match_score": matches[0]["match_score"] if matches else 0,
    }

# COMMAND ----------

# DBTITLE 1,End-to-End Agent Function
@mlflow.trace(name="generate_plan_from_user_input", span_type="AGENT")
def generate_plan_from_user_input(
    user_input: str,
    model_endpoint: Optional[str] = None,
    review_model_endpoint: Optional[str] = None,
    return_context: bool = True,
) -> Dict[str, Any]:
    """
    Complete end-to-end flow:
    1. Extract user profile from natural language
    2. Validate safety and eligibility
    3. Match against athlete profiles
    4. Retrieve exercises, benchmarks, nutrition
    5. Generate personalized coaching plan
    """
    started_at = time.time()
    endpoint = model_endpoint or MODEL_ENDPOINT
    reviewer_endpoint = review_model_endpoint or REVIEW_MODEL_ENDPOINT
    
    # Tag trace with user input
    try:
        mlflow.update_current_trace(
            tags={
                "user_input_preview": user_input[:100] + "..." if len(user_input) > 100 else user_input,
                "model_endpoint": endpoint,
                "review_model_endpoint": reviewer_endpoint,
            }
        )
    except Exception:
        pass
    
    # Step 1: Extract profile
    try:
        extracted_profile = extract_user_profile(user_input, model_endpoint=endpoint)
    except Exception as exc:
        return {
            "status": "error",
            "error": f"Profile extraction failed: {exc}",
            "user_input": user_input,
            "latency_seconds": round(time.time() - started_at, 3),
        }
    
    # Tag trace with extracted profile
    try:
        mlflow.update_current_trace(
            tags={
                "extracted_age": str(extracted_profile.get("age", "unknown")),
                "extracted_goal": str(extracted_profile.get("primary_goal", "unknown"))[:50],
            }
        )
    except Exception:
        pass
    
    # Step 2: Validate profile
    validation = validate_user_profile(extracted_profile)
    
    # Tag trace with validation status
    try:
        mlflow.update_current_trace(
            tags={
                "validation_status": "allowed" if validation["allowed"] else "rejected",
                "age_group": validation.get("age_group", "unknown"),
            }
        )
    except Exception:
        pass
    
    if not validation["allowed"]:
        # Return rejection or clarification request
        result = {
            "status": "rejected",
            "reason": validation["reason"],
            "validation": validation,
            "extracted_profile": extracted_profile,
            "user_input": user_input,
            "latency_seconds": round(time.time() - started_at, 3),
        }
        
        # If missing fields, provide clarifying questions
        if validation.get("clarifying_questions"):
            result["status"] = "needs_clarification"
            questions = validation["clarifying_questions"]
            # Ensure questions is a list and join them into a string
            if isinstance(questions, list):
                result["response"] = (
                    "I need a bit more information to create your plan. "
                    + " ".join(str(q) for q in questions if q)
                )
            else:
                result["response"] = (
                    "I need a bit more information to create your plan. "
                    + str(questions)
                )
        else:
            # Use rejection response if provided - ensure it's always a string
            resp = validation.get("response", validation["reason"])
            result["response"] = str(resp) if not isinstance(resp, str) else resp
        
        return result
    
    # Step 3: Match profile
    try:
        matching_result = match_user_profile(extracted_profile)
        top_matches = matching_result["top_matches"]
        archetype = matching_result["archetype"]
    except Exception as exc:
        return {
            "status": "error",
            "error": f"Profile matching failed: {exc}",
            "extracted_profile": extracted_profile,
            "validation": validation,
            "user_input": user_input,
            "latency_seconds": round(time.time() - started_at, 3),
        }
    
    # Tag trace with match info
    try:
        if top_matches:
            mlflow.update_current_trace(
                tags={
                    "top_match_id": top_matches[0]["athlete_id"],
                    "match_score": str(round(top_matches[0]["match_score"], 2)),
                    "archetype": archetype[:100],
                }
            )
    except Exception:
        pass
    
    # Step 4: Build enriched profile for tool context
    # Use the top matched athlete profile as a base, but override with user specifics
    if top_matches:
        base_profile = top_matches[0]["athlete_profile"].copy()
    else:
        base_profile = {}
    
    # Create enriched profile
    enriched_profile = {
        "athlete_id": f"user_{int(started_at)}",
        "age": extracted_profile.get("age"),
        "age_group": validation.get("age_group"),
        "sport": extracted_profile.get("sport", "basketball"),
        "position": extracted_profile.get("position") or base_profile.get("position", "unknown"),
        "goal": extracted_profile.get("primary_goal"),
        "injury_status": extracted_profile.get("injury_status", "none"),
        "available_days": extracted_profile.get("training_days_available", 3),
        "equipment": extracted_profile.get("equipment_access") or base_profile.get("equipment", "bodyweight"),
        "experience_level": extracted_profile.get("skill_level") or base_profile.get("experience_level", "intermediate"),
        "user_notes": extracted_profile.get("free_text_notes", ""),
    }
    
    # Step 5: Build tool context (retrieve exercises, benchmarks, nutrition)
    context = build_tool_context(enriched_profile, user_input)
    
    if not context.get("safety", {}).get("allowed", True):
        return {
            "status": "rejected",
            "reason": context["safety"]["reason"],
            "response": context["safety"]["response"],
            "extracted_profile": extracted_profile,
            "validation": validation,
            "matching": matching_result,
            "latency_seconds": round(time.time() - started_at, 3),
        }
    
    # Step 6: Generate coaching plan
    user_payload = {
        "athlete_profile": enriched_profile,
        "user_input": user_input,
        "extracted_profile": extracted_profile,
        "matched_archetype": archetype,
        "top_matches": [
            {
                "athlete_id": m["athlete_id"],
                "match_score": m["match_score"],
                "age": m["athlete_profile"].get("age"),
                "position": m["athlete_profile"].get("position"),
                "goal": m["athlete_profile"].get("goal"),
            }
            for m in top_matches[:3]
        ],
        "tool_context": context,
        "format_reminder": (
            "Return a safe weekly plan with a safety note, Today / This Week summary, daily plan table, "
            "nutrition guidance, recovery guidance, rationale, and progress metrics. "
            f"Explain that this plan is based on matching with similar {archetype} athletes."
        ),
    }
    
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, indent=2)},
    ]
    
    response_text = call_llm(messages, endpoint, profile=enriched_profile, context=context)
    plan_review = review_plan_with_second_llm(
        draft_response=response_text,
        profile=enriched_profile,
        context=context,
        review_model_endpoint=reviewer_endpoint,
    )
    final_status = "answered" if plan_review.get("passes", False) else "needs_human_review"
    
    result = {
        "status": final_status,
        "plan_id": f"user_{int(started_at)}",
        "user_input": user_input,
        "extracted_profile": extracted_profile,
        "validation": validation,
        "matching": matching_result,
        "enriched_profile": enriched_profile,
        "response": response_text,
        "plan_review": plan_review,
        "safety": context["safety"],
        "model_endpoint": endpoint,
        "review_model_endpoint": reviewer_endpoint,
        "latency_seconds": round(time.time() - started_at, 3),
    }
    
    if return_context:
        result["tool_context"] = context
    
    # Final trace tags
    try:
        response_preview = response_text[:200] + "..." if len(response_text) > 200 else response_text
        mlflow.update_current_trace(
            tags={
                "status": result["status"],
                "response_preview": response_preview,
                "case_type": result["status"],
                "review_passes": str(plan_review.get("passes", False)),
                "review_requires_human_review": str(plan_review.get("requires_human_review", False)),
            }
        )
    except Exception:
        pass
    
    return result

# COMMAND ----------

def extract_chat_content(response: Any) -> str:
    if hasattr(response, "to_dict"):
        response = response.to_dict()
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            return message.get("content") or choices[0].get("text") or json.dumps(response)
        if "predictions" in response:
            return json.dumps(response["predictions"], indent=2)
    return str(response)


def mock_llm_response(profile: Dict[str, Any], context: Dict[str, Any]) -> str:
    days = context["safety"].get("safe_training_days", 3)
    exercises = context.get("recommended_exercises", [])[:3]
    exercise_names = ", ".join([item.get("exercise_name", "general movement") for item in exercises])
    injury_status = profile.get("injury_status", "none")
    age_group = context["safety"].get("age_group", "youth")
    audience_note = "parent or coach oversight" if age_group == "youth" else "self-directed or coach-supported review"
    injury_note = "Because an injury is listed, keep all work low impact and confirm with a coach or clinician." if injury_status != "none" else "No injury is listed, but warmups and recovery still matter."

    return f"""
Safety note: This is educational basketball guidance for an athlete age {profile.get('age')} with {audience_note}. {injury_note}

Today / This Week: Focus on {profile.get('goal')} for {days} planned training days, with one clear recovery check after every session.

| Day | Focus | Plan |
| --- | --- | --- |
| Day 1 | Skill and movement | Dynamic warmup, basketball skill work tied to {profile.get('goal')}, light strength using {exercise_names or 'bodyweight basics'}, cooldown stretch. |
| Day 2 | Recovery | Mobility, easy walk or form shooting, hydration check, and sleep target. |
| Day 3 | Basketball development | Skill block, short conditioning segment, core stability, cooldown. |
| Day 4 | Recovery or light practice | Low-intensity ball handling, stretching, and soreness check. |
| Day 5 | Strength and quality | Strength and movement quality session if the athlete feels good. |

Training volume: Use {days} planned training days this week. Do not add extra intense sessions just because a day feels easy.

Why this fits: The plan uses the athlete profile, selected exercise themes, safety rules, and basketball benchmark context to keep the work specific but conservative.

Nutrition guidance: Use regular meals with carbohydrates, protein-rich foods, fruits or vegetables, and water. Avoid supplements, fasting, and weight-cutting.

Progress metrics for next week:
- sessions_completed
- pain_or_soreness_flag
- sleep_quality
- hydration_check
- skill_confidence
- coach_parent_reviewed
""".strip()


def heuristic_plan_review(draft_response: str, profile: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    response = (draft_response or "").lower()
    safety = context.get("safety", {})
    revision_requests = []
    banned_terms = ["creatine", "fasting plan", "weight cut", "steroid", "diagnose", "prescribe"]

    if any(term in response for term in banned_terms):
        revision_requests.append("Remove supplements, fasting, weight-cutting, or medical-treatment language.")
    if "recovery" not in response:
        revision_requests.append("Add recovery guidance.")
    if "hydration" not in response:
        revision_requests.append("Add hydration guidance.")
    if "sleep" not in response:
        revision_requests.append("Add sleep guidance.")
    if (profile.get("injury_status") or "none").lower() != "none" and not any(term in response for term in ["low impact", "medical review", "clinician", "coach"]):
        revision_requests.append("Address the injury note with lower-impact work and coach or medical review.")
    if safety.get("age_group") == "youth" and "parent" not in response and "coach" not in response:
        revision_requests.append("Mention parent or coach oversight for youth athletes.")

    passes = len(revision_requests) == 0
    safety_score = 5 if passes else max(1, 5 - len(revision_requests))
    grounding_score = 5 if "exercise" in response or "benchmark" in response or "tool" in response else 3
    clarity_score = 5 if "day" in response and ("|" in response or "-" in response) else 3

    return {
        "passes": passes,
        "safety_score": safety_score,
        "grounding_score": grounding_score,
        "clarity_score": clarity_score,
        "requires_human_review": (not passes) or (profile.get("injury_status") or "none").lower() != "none",
        "reviewer_notes": "Heuristic reviewer used because the review LLM endpoint is not configured.",
        "revision_requests": revision_requests,
    }


@mlflow.trace(name="call_llm", span_type="LLM")
def call_llm(messages: List[Dict[str, str]], model_endpoint: str, profile=None, context=None) -> str:
    if ALLOW_MOCK_LLM or not model_endpoint or model_endpoint.startswith("TODO_"):
        return mock_llm_response(profile or {}, context or {})

    import mlflow.deployments

    client = mlflow.deployments.get_deploy_client("databricks")
    response = client.predict(
        endpoint=model_endpoint,
        inputs={
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1400,
        },
    )
    return extract_chat_content(response)


@mlflow.trace(name="review_plan_with_second_llm", span_type="LLM")
def review_plan_with_second_llm(
    draft_response: str,
    profile: Dict[str, Any],
    context: Dict[str, Any],
    review_model_endpoint: Optional[str] = None,
) -> Dict[str, Any]:
    endpoint = review_model_endpoint or REVIEW_MODEL_ENDPOINT
    if ALLOW_MOCK_LLM or not endpoint or endpoint.startswith("TODO_"):
        review = heuristic_plan_review(draft_response, profile, context)
        review["review_model_endpoint"] = endpoint
        return review

    review_payload = {
        "athlete_profile": profile,
        "tool_context": context,
        "draft_response": draft_response,
    }
    messages = [
        {"role": "system", "content": PLAN_REVIEW_PROMPT},
        {"role": "user", "content": json.dumps(review_payload, indent=2)},
    ]

    import mlflow.deployments

    client = mlflow.deployments.get_deploy_client("databricks")
    response = client.predict(
        endpoint=endpoint,
        inputs={
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 700,
        },
    )
    response_text = extract_chat_content(response)

    try:
        json_match = re.search(r"```(?:json)?\s*({.*?})\s*```", response_text, re.DOTALL)
        review = json.loads(json_match.group(1) if json_match else response_text)
    except Exception as exc:
        review = heuristic_plan_review(draft_response, profile, context)
        review["reviewer_notes"] = f"{review['reviewer_notes']} Review LLM response could not be parsed: {exc}"

    review.setdefault("passes", False)
    review.setdefault("safety_score", 0)
    review.setdefault("grounding_score", 0)
    review.setdefault("clarity_score", 0)
    review.setdefault("requires_human_review", not bool(review.get("passes")))
    review.setdefault("reviewer_notes", "")
    review.setdefault("revision_requests", [])
    review["review_model_endpoint"] = endpoint
    return review


@mlflow.trace(name="generate_weekly_plan", span_type="AGENT")
def generate_weekly_plan(
    athlete_id: str,
    request: str,
    model_endpoint: Optional[str] = None,
    review_model_endpoint: Optional[str] = None,
    return_context: bool = False,
) -> Dict[str, Any]:
    started_at = time.time()
    endpoint = model_endpoint or MODEL_ENDPOINT
    reviewer_endpoint = review_model_endpoint or REVIEW_MODEL_ENDPOINT
    profile = get_athlete_profile(athlete_id)
    context = build_tool_context(profile, request)
    
    # Set trace tags
    try:
        mlflow.update_current_trace(
            tags={
                "athlete_id": athlete_id,
                "age_group": profile.get("age_group", "unknown"),
                "model_endpoint": endpoint,
                "review_model_endpoint": reviewer_endpoint,
                "request_preview": request[:100] + "..." if len(request) > 100 else request,
            }
        )
    except Exception:
        pass  # Don't fail if trace tagging fails

    if not context["safety"]["allowed"]:
        result = {
            "status": "rejected",
            "athlete_id": athlete_id,
            "model_endpoint": endpoint,
            "request": request,
            "response": context["safety"]["response"],
            "safety": context["safety"],
            "latency_seconds": round(time.time() - started_at, 3),
        }
        if return_context:
            result["tool_context"] = context
        return result

    user_payload = {
        "athlete_profile": profile,
        "user_request": request,
        "tool_context": context,
        "format_reminder": (
            "Return a safe weekly plan with a safety note, Today / This Week summary, daily plan table, "
            "nutrition guidance, recovery guidance, rationale, and progress metrics."
        ),
    }
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, indent=2)},
    ]
    response_text = call_llm(messages, endpoint, profile=profile, context=context)
    plan_review = review_plan_with_second_llm(
        draft_response=response_text,
        profile=profile,
        context=context,
        review_model_endpoint=reviewer_endpoint,
    )
    final_status = "answered" if plan_review.get("passes", False) else "needs_human_review"

    result = {
        "status": final_status,
        "plan_id": f"{athlete_id}-{int(started_at)}",
        "athlete_id": athlete_id,
        "model_endpoint": endpoint,
        "review_model_endpoint": reviewer_endpoint,
        "request": request,
        "response": response_text,
        "plan_review": plan_review,
        "safety": context["safety"],
        "latency_seconds": round(time.time() - started_at, 3),
    }
    if return_context:
        result["tool_context"] = context
    
    # Update trace with response tags
    try:
        response_preview = response_text[:200] + "..." if len(response_text) > 200 else response_text
        mlflow.update_current_trace(
            tags={
                "status": result["status"],
                "response_preview": response_preview,
                "age_group": context["safety"].get("age_group", "unknown"),
                "case_type": result["status"],
                "review_passes": str(plan_review.get("passes", False)),
                "review_requires_human_review": str(plan_review.get("requires_human_review", False)),
            }
        )
    except Exception:
        pass  # Don't fail the request if trace update fails
    
    return result

# COMMAND ----------

RUN_DEMO = _widget("run_demo", "false").lower() == "true"

if RUN_DEMO:
    demo_result = generate_weekly_plan(
        athlete_id="A001",
        request="Create a safe weekly basketball plan focused on the athlete's main goal.",
        return_context=True,
    )
    print(json.dumps(demo_result, indent=2))

# COMMAND ----------

# DBTITLE 1,Model Comparison Section
# MAGIC %md
# MAGIC ## Model Comparison
# MAGIC
# MAGIC This section compares the two LLMs used in the Elite Athletic AI Advisor:
# MAGIC
# MAGIC 1. **Main Agent Model**: `databricks-gpt-oss-120b` - Generates weekly training plans
# MAGIC 2. **Review Model**: `databricks-llama-4-maverick` - Reviews plans for safety and quality
# MAGIC
# MAGIC We'll test both models by:
# MAGIC - Generating plans for the same athlete profile
# MAGIC - Comparing response quality, safety adherence, and latency
# MAGIC - Showing how the review model evaluates each plan

# COMMAND ----------

# DBTITLE 1,Model Comparison: Setup and Test Profile
import time
from datetime import datetime

# Test athlete profile for model comparison
test_profile = {
    "athlete_id": "TEST001",
    "age": 16,
    "sport": "basketball",
    "position": "Point Guard",
    "goal": "Improve ball handling and speed",
    "experience_level": "intermediate",
    "injury_status": "minor ankle soreness",
    "available_days": 4,
    "equipment": "Basketball, Cones, Gym Access",
    "age_group": "youth"
}

test_request = "Create a safe weekly basketball plan focused on improving ball handling and speed, while being mindful of ankle soreness."

print("=" * 80)
print("MODEL COMPARISON TEST SETUP")
print("=" * 80)
print(f"\nTest Profile: {test_profile['athlete_id']}")
print(f"Age: {test_profile['age']} ({test_profile['age_group']})")
print(f"Position: {test_profile['position']}")
print(f"Goal: {test_profile['goal']}")
print(f"Injury: {test_profile['injury_status']}")
print(f"\nRequest: {test_request}")
print("\n" + "=" * 80)

# COMMAND ----------

# DBTITLE 1,Model Comparison: Test Model 1 (GPT-OSS-120B)
print("\n\n" + "=" * 80)
print("MODEL 1: databricks-gpt-oss-120b (Main Agent Model)")
print("=" * 80)

# Build context for the test
context_model1 = build_tool_context(test_profile, test_request)

# Prepare messages for Model 1
user_payload_m1 = {
    "athlete_profile": test_profile,
    "user_request": test_request,
    "tool_context": context_model1,
    "format_reminder": (
        "Return a safe weekly plan with a safety note, Today / This Week summary, daily plan table, "
        "nutrition guidance, recovery guidance, rationale, and progress metrics."
    ),
}

messages_m1 = [
    {"role": "system", "content": AGENT_SYSTEM_PROMPT},
    {"role": "user", "content": json.dumps(user_payload_m1, indent=2)},
]

# Call Model 1
start_time_m1 = time.time()
response_m1 = call_llm(messages_m1, MODEL_ENDPOINT, profile=test_profile, context=context_model1)
latency_m1 = round(time.time() - start_time_m1, 3)

print(f"\n✓ Model: {MODEL_ENDPOINT}")
print(f"✓ Latency: {latency_m1} seconds")
print(f"✓ Response Length: {len(response_m1)} characters")
print(f"\nResponse Preview (first 600 chars):")
print("-" * 80)
print(response_m1[:600] + "..." if len(response_m1) > 600 else response_m1)
print("-" * 80)

# Store for comparison
model1_result = {
    "model": MODEL_ENDPOINT,
    "response": response_m1,
    "latency": latency_m1,
    "response_length": len(response_m1)
}

# COMMAND ----------

# DBTITLE 1,Model Comparison: Test Model 2 (Llama-4-Maverick)
print("\n\n" + "=" * 80)
print("MODEL 2: databricks-llama-4-maverick (Review Model)")
print("=" * 80)
print("Testing this model as a PLAN GENERATOR (normally it's used for review)")
print("=" * 80)

# Use same context for fair comparison
context_model2 = context_model1

# Prepare messages for Model 2 (using same prompt)
user_payload_m2 = {
    "athlete_profile": test_profile,
    "user_request": test_request,
    "tool_context": context_model2,
    "format_reminder": (
        "Return a safe weekly plan with a safety note, Today / This Week summary, daily plan table, "
        "nutrition guidance, recovery guidance, rationale, and progress metrics."
    ),
}

messages_m2 = [
    {"role": "system", "content": AGENT_SYSTEM_PROMPT},
    {"role": "user", "content": json.dumps(user_payload_m2, indent=2)},
]

# Call Model 2
start_time_m2 = time.time()
response_m2 = call_llm(messages_m2, REVIEW_MODEL_ENDPOINT, profile=test_profile, context=context_model2)
latency_m2 = round(time.time() - start_time_m2, 3)

print(f"\n✓ Model: {REVIEW_MODEL_ENDPOINT}")
print(f"✓ Latency: {latency_m2} seconds")
print(f"✓ Response Length: {len(response_m2)} characters")
print(f"\nResponse Preview (first 600 chars):")
print("-" * 80)
print(response_m2[:600] + "..." if len(response_m2) > 600 else response_m2)
print("-" * 80)

# Store for comparison
model2_result = {
    "model": REVIEW_MODEL_ENDPOINT,
    "response": response_m2,
    "latency": latency_m2,
    "response_length": len(response_m2)
}

# COMMAND ----------

# DBTITLE 1,Model Comparison: Review Both Plans
print("\n\n" + "=" * 80)
print("PLAN REVIEWS: How does the review model evaluate each plan?")
print("=" * 80)

# Review Model 1's plan
print("\n" + "-" * 80)
print("Reviewing Model 1 (GPT-OSS-120B) Plan:")
print("-" * 80)
review_m1 = review_plan_with_second_llm(
    draft_response=response_m1,
    profile=test_profile,
    context=context_model1,
    review_model_endpoint=REVIEW_MODEL_ENDPOINT
)
print(f"✓ Passes: {review_m1.get('passes', False)}")
print(f"✓ Safety Score: {review_m1.get('safety_score', 0)}/5")
print(f"✓ Grounding Score: {review_m1.get('grounding_score', 0)}/5")
print(f"✓ Clarity Score: {review_m1.get('clarity_score', 0)}/5")
print(f"✓ Requires Human Review: {review_m1.get('requires_human_review', False)}")
if review_m1.get('revision_requests'):
    print(f"\nRevision Requests:")
    for req in review_m1['revision_requests']:
        print(f"  - {req}")

# Review Model 2's plan
print("\n" + "-" * 80)
print("Reviewing Model 2 (Llama-4-Maverick) Plan:")
print("-" * 80)
review_m2 = review_plan_with_second_llm(
    draft_response=response_m2,
    profile=test_profile,
    context=context_model2,
    review_model_endpoint=REVIEW_MODEL_ENDPOINT
)
print(f"✓ Passes: {review_m2.get('passes', False)}")
print(f"✓ Safety Score: {review_m2.get('safety_score', 0)}/5")
print(f"✓ Grounding Score: {review_m2.get('grounding_score', 0)}/5")
print(f"✓ Clarity Score: {review_m2.get('clarity_score', 0)}/5")
print(f"✓ Requires Human Review: {review_m2.get('requires_human_review', False)}")
if review_m2.get('revision_requests'):
    print(f"\nRevision Requests:")
    for req in review_m2['revision_requests']:
        print(f"  - {req}")

# COMMAND ----------

# DBTITLE 1,Model Comparison: Summary Table
print("\n\n" + "=" * 80)
print("SIDE-BY-SIDE COMPARISON SUMMARY")
print("=" * 80)

# Calculate quality metrics
total_score_m1 = review_m1.get('safety_score', 0) + review_m1.get('grounding_score', 0) + review_m1.get('clarity_score', 0)
total_score_m2 = review_m2.get('safety_score', 0) + review_m2.get('grounding_score', 0) + review_m2.get('clarity_score', 0)

# Build comparison table
print(f"\n{'Metric':<30} {'Model 1 (GPT-OSS)':<25} {'Model 2 (Llama-4)':<25}")
print("=" * 80)
print(f"{'Model Name':<30} {MODEL_ENDPOINT[:24]:<25} {REVIEW_MODEL_ENDPOINT[:24]:<25}")
print(f"{'Latency (seconds)':<30} {latency_m1:<25} {latency_m2:<25}")
print(f"{'Response Length (chars)':<30} {len(response_m1):<25} {len(response_m2):<25}")
print(f"{'Review Passes':<30} {str(review_m1.get('passes', False)):<25} {str(review_m2.get('passes', False)):<25}")
print(f"{'Safety Score (out of 5)':<30} {review_m1.get('safety_score', 0):<25} {review_m2.get('safety_score', 0):<25}")
print(f"{'Grounding Score (out of 5)':<30} {review_m1.get('grounding_score', 0):<25} {review_m2.get('grounding_score', 0):<25}")
print(f"{'Clarity Score (out of 5)':<30} {review_m1.get('clarity_score', 0):<25} {review_m2.get('clarity_score', 0):<25}")
print(f"{'Total Quality Score (out of 15)':<30} {total_score_m1:<25} {total_score_m2:<25}")
print(f"{'Requires Human Review':<30} {str(review_m1.get('requires_human_review', False)):<25} {str(review_m2.get('requires_human_review', False)):<25}")

print("\n" + "=" * 80)
print("KEY FINDINGS:")
print("=" * 80)

# Determine winner
if latency_m1 < latency_m2:
    print(f"⚡ Faster Response: Model 1 (GPT-OSS-120B) by {latency_m2 - latency_m1:.3f}s")
else:
    print(f"⚡ Faster Response: Model 2 (Llama-4-Maverick) by {latency_m1 - latency_m2:.3f}s")

if total_score_m1 > total_score_m2:
    print(f"🏆 Higher Quality: Model 1 (GPT-OSS-120B) with score {total_score_m1}/15")
elif total_score_m2 > total_score_m1:
    print(f"🏆 Higher Quality: Model 2 (Llama-4-Maverick) with score {total_score_m2}/15")
else:
    print(f"🤝 Equal Quality: Both models scored {total_score_m1}/15")

if len(response_m1) > len(response_m2):
    print(f"📝 More Detailed: Model 1 (GPT-OSS-120B) with {len(response_m1) - len(response_m2)} more characters")
elif len(response_m2) > len(response_m1):
    print(f"📝 More Detailed: Model 2 (Llama-4-Maverick) with {len(response_m2) - len(response_m1)} more characters")
else:
    print(f"📝 Equal Detail: Both responses have {len(response_m1)} characters")

print("\n" + "=" * 80)
print("NOTE: Currently using mock LLM mode. Set allow_mock_llm=false to test real endpoints.")
print("=" * 80)

# COMMAND ----------

# DBTITLE 1,Tool Use Demo Section
# MAGIC %md
# MAGIC ## Tool Use Demo
# MAGIC
# MAGIC This section demonstrates how the agent uses each tool to gather context before generating a weekly plan. We'll run one example for athlete A001 and show:
# MAGIC
# MAGIC 1. Athlete profile lookup
# MAGIC 2. Safety check evaluation
# MAGIC 3. Exercise retrieval
# MAGIC 4. Basketball benchmark lookup
# MAGIC 5. Nutrition guidance lookup
# MAGIC 6. Final generated weekly plan
# MAGIC
# MAGIC This makes it easy to show the team how the agent actually uses tools before generating the answer.

# COMMAND ----------

# DBTITLE 1,Tool Use Demo: Athlete Profile
# Demo: Show tool usage for athlete A001
demo_athlete_id = "A001"
demo_request = "Create a safe weekly basketball plan focused on the athlete's main goal."

print("=" * 80)
print("TOOL USE DEMO: Athlete Profile Lookup")
print("=" * 80)
profile = get_athlete_profile(demo_athlete_id)
print(json.dumps(profile, indent=2))

# COMMAND ----------

# DBTITLE 1,Tool Use Demo: Safety Check
print("\n" + "=" * 80)
print("TOOL USE DEMO: Safety Check")
print("=" * 80)
safety_result = check_safety(profile, demo_request)
print(json.dumps(safety_result, indent=2))

# COMMAND ----------

# DBTITLE 1,Tool Use Demo: Exercise Retrieval
print("\n" + "=" * 80)
print("TOOL USE DEMO: Retrieve Exercises")
print("=" * 80)
exercises = retrieve_exercises(
    goal=profile.get("goal", ""),
    equipment=profile.get("equipment", ""),
    injury_status=profile.get("injury_status", "none"),
)
for i, ex in enumerate(exercises, 1):
    print(f"\n{i}. {ex.get('exercise_name', 'N/A')}")
    print(f"   Equipment: {ex.get('equipment', 'N/A')}")
    print(f"   Goal Tag: {ex.get('goal_tag', 'N/A')}")

# COMMAND ----------

# DBTITLE 1,Tool Use Demo: Basketball Benchmark
print("\n" + "=" * 80)
print("TOOL USE DEMO: Basketball Benchmark Lookup")
print("=" * 80)
benchmarks = get_basketball_benchmark(profile.get("position", ""))
for i, bench in enumerate(benchmarks, 1):
    print(f"\n{i}. Position: {bench.get('position', 'N/A')}")
    print(f"   Points per game: {bench.get('avg_points', 'N/A')}")
    print(f"   Assists per game: {bench.get('avg_assists', 'N/A')}")
    print(f"   Blocks per game: {bench.get('avg_blocks', 'N/A')}")

# COMMAND ----------

# DBTITLE 1,Tool Use Demo: Nutrition Lookup
print("\n" + "=" * 80)
print("TOOL USE DEMO: Nutrition Guidance Lookup")
print("=" * 80)
nutrition = lookup_nutrition(
    goal=profile.get("goal", ""),
    injury_status=profile.get("injury_status", "none"),
)
if nutrition:
    for i, nutr in enumerate(nutrition, 1):
        print(f"\n{i}. Goal Tag: {nutr.get('goal_tag', 'N/A')}")
        print(f"   Timing: {nutr.get('timing', 'N/A')}")
        print(f"   Guidance: {nutr.get('guidance_text', 'N/A')[:100]}...")
else:
    print("\nNo nutrition guidance records available.")
    print("(The nutrition guidance table will be populated in a future update.)")

# COMMAND ----------

# DBTITLE 1,Tool Use Demo: Nutrient Lookup
print("\n" + "=" * 80)
print("TOOL USE DEMO: Nutrient Lookup")
print("=" * 80)
print("Searching for protein-rich foods...\n")

# Demo: Search for protein nutrients
protein_nutrients = lookup_nutrient("protein", limit=10)

if protein_nutrients:
    for i, nutrient in enumerate(protein_nutrients, 1):
        print(f"{i}. Nutrient: {nutrient.get('nutrient_name', 'N/A')}")
        print(f"   ID: {nutrient.get('food_nutrient_id', 'N/A')}")
        print(f"   Amount: {nutrient.get('adjusted_amount', 'N/A')}")
        print(f"   Lab Method: {nutrient.get('lab_method_id', 'N/A')}")
        print()
    
    print("-" * 80)
    print(f"Total results: {len(protein_nutrients)} nutrients found")
else:
    print("No protein nutrients found in the database.")

# COMMAND ----------

# DBTITLE 1,User Profile Intake Demo Section
# MAGIC %md
# MAGIC ## User Profile Intake Demo
# MAGIC
# MAGIC This section demonstrates the complete user profile intake flow where new users can describe themselves in natural language and receive personalized coaching.
# MAGIC
# MAGIC For each example, we show:
# MAGIC 1. Raw user input (natural language)
# MAGIC 2. Extracted structured profile
# MAGIC 3. Validation result
# MAGIC 4. Top 3 profile matches with scores
# MAGIC 5. Final coaching response
# MAGIC
# MAGIC This demonstrates how the agent transforms from requiring pre-existing `athlete_id` to accepting free-form user input.

# COMMAND ----------

# DBTITLE 1,Intake Demo: Youth with Complete Profile
# Example 1: Youth athlete with complete profile
print("\n" + "=" * 80)
print("USER PROFILE INTAKE DEMO #1: Youth with Complete Profile")
print("=" * 80)

user_input_1 = """
I'm 16 years old and play shooting guard on my high school varsity team. I've been 
playing basketball for 5 years. I want to improve my explosiveness and ball handling 
for upcoming tryouts next month. I have minor knee soreness sometimes from a previous 
sprain, but nothing serious. I'm about 6'1" and 170 lbs, pretty lean build. I sleep 
about 7 hours on school nights. I can train 4 days a week during off-season. I have 
access to dumbbells and a driveway hoop at home. I'm right-handed and would say my 
fitness level is pretty athletic.
"""

print("\nRAW USER INPUT:")
print("-" * 80)
print(user_input_1.strip())

result_1 = generate_plan_from_user_input(
    user_input=user_input_1,
    return_context=True,
)

print("\n" + "-" * 80)
print("EXTRACTED PROFILE:")
print("-" * 80)
print(json.dumps(result_1.get("extracted_profile", {}), indent=2))

print("\n" + "-" * 80)
print("VALIDATION RESULT:")
print("-" * 80)
validation = result_1.get("validation", {})
print(f"Status: {'✓ ALLOWED' if validation.get('allowed') else '✗ REJECTED'}")
print(f"Age Group: {validation.get('age_group', 'N/A')}")
print(f"Oversight: {validation.get('oversight_required', 'N/A')}")
if validation.get("warnings"):
    print(f"Warnings: {', '.join(validation['warnings'])}")

print("\n" + "-" * 80)
print("TOP 3 PROFILE MATCHES:")
print("-" * 80)
matching = result_1.get("matching", {})
for i, match in enumerate(matching.get("top_matches", [])[:3], 1):
    athlete = match["athlete_profile"]
    print(f"\n{i}. Athlete {match['athlete_id']} (Score: {match['match_score']})")
    print(f"   Age: {athlete.get('age')}, Position: {athlete.get('position')}")
    print(f"   Goal: {athlete.get('goal')}")
    print(f"   Injury: {athlete.get('injury_status', 'none')}")

print(f"\nRecommended Archetype: {matching.get('archetype', 'N/A')}")

print("\n" + "-" * 80)
print("FINAL COACHING RESPONSE:")
print("-" * 80)
print(result_1.get("response", "No response generated")[:800] + "...")
print(f"\nStatus: {result_1['status']}")
print(f"Latency: {result_1['latency_seconds']} seconds")

# COMMAND ----------

# DBTITLE 1,Intake Demo: Youth with Injury Concern
# Example 2: Youth athlete with injury concern
print("\n\n" + "=" * 80)
print("USER PROFILE INTAKE DEMO #2: Youth with Injury Concern")
print("=" * 80)

user_input_2 = """
Hey, I'm 15 and play point guard on my middle school team. Been playing for 2 years 
and still pretty new to competitive basketball. I'm recovering from an ankle sprain 
from last month and want to work on my conditioning and ball handling without jumping 
too much. I'm about 5'8" and 140 lbs, average build. I sleep around 8 hours most nights. 
Can train 3 days per week during pre-season. Have access to a local gym. Left-handed. 
I'd say my fitness level is moderate - still building up my endurance.
"""

print("\nRAW USER INPUT:")
print("-" * 80)
print(user_input_2.strip())

result_2 = generate_plan_from_user_input(
    user_input=user_input_2,
    return_context=True,
)

print("\n" + "-" * 80)
print("EXTRACTED PROFILE:")
print("-" * 80)
extracted = result_2.get("extracted_profile", {})
print(f"Age: {extracted.get('age')}")
print(f"Position: {extracted.get('position')}")
print(f"Primary Goal: {extracted.get('primary_goal')}")
print(f"Injury Status: {extracted.get('injury_status')}")
print(f"Training Days: {extracted.get('training_days_available')}")

print("\n" + "-" * 80)
print("VALIDATION & MATCHING:")
print("-" * 80)
validation_2 = result_2.get("validation", {})
matching_2 = result_2.get("matching", {})
print(f"Validation: {'✓ ALLOWED' if validation_2.get('allowed') else '✗ REJECTED'}")
print(f"Archetype: {matching_2.get('archetype', 'N/A')}")
print(f"Top Match: {matching_2.get('top_matches', [{}])[0].get('athlete_id', 'N/A')} "
      f"(Score: {matching_2.get('top_matches', [{}])[0].get('match_score', 0)})")

if validation_2.get("warnings"):
    print(f"\nSafety Warnings:")
    for warning in validation_2["warnings"]:
        print(f"  - {warning}")

print("\n" + "-" * 80)
print("COACHING RESPONSE (Preview):")
print("-" * 80)
print(result_2.get("response", "No response")[:600] + "...")

# COMMAND ----------

# DBTITLE 1,Intake Demo: Adult 18+ Preview
# Example 3: Adult 18+ preview user
print("\n\n" + "=" * 80)
print("USER PROFILE INTAKE DEMO #3: Adult 18+ Preview User")
print("=" * 80)

user_input_3 = """
I'm 19 and play recreational basketball at the local YMCA. Been playing pickup for 
about 3 years since high school. I'm a forward, about 6'3" and 195 lbs, pretty muscular 
build. Want to improve my conditioning and strength for competitive pickup games. Sleep 
about 6-7 hours with work schedule. Can work out 5 days a week during off-season and 
have a full gym membership with weights and court access. Right-handed. Would say my 
fitness level is athletic but want to get more consistent.
"""

print("\nRAW USER INPUT:")
print("-" * 80)
print(user_input_3.strip())

result_3 = generate_plan_from_user_input(
    user_input=user_input_3,
    return_context=True,
)

print("\n" + "-" * 80)
print("KEY RESULTS:")
print("-" * 80)
extracted_3 = result_3.get("extracted_profile", {})
validation_3 = result_3.get("validation", {})
matching_3 = result_3.get("matching", {})

print(f"Age: {extracted_3.get('age')} ({validation_3.get('age_group', 'N/A')} classification)")
print(f"Primary Goal: {extracted_3.get('primary_goal')}")
print(f"Archetype: {matching_3.get('archetype', 'N/A')}")
print(f"Status: {result_3['status']}")

if validation_3.get("age_group") == "adult":
    print("\n✓ Adult preview mode activated")
    print(f"  Max training days: {validation_3.get('max_training_days')}")
    print(f"  Oversight: {validation_3.get('oversight_required')}")

print("\n" + "-" * 80)
print("RESPONSE PREVIEW:")
print("-" * 80)
print(result_3.get("response", "No response")[:500] + "...")

# COMMAND ----------

# DBTITLE 1,Intake Demo: Missing Information
# Example 4: Missing information requiring clarification
print("\n\n" + "=" * 80)
print("USER PROFILE INTAKE DEMO #4: Missing Information")
print("=" * 80)

user_input_4 = """
I play basketball and want to get better. Help me create a training plan.
"""

print("\nRAW USER INPUT:")
print("-" * 80)
print(user_input_4.strip())

result_4 = generate_plan_from_user_input(
    user_input=user_input_4,
    return_context=True,
)

print("\n" + "-" * 80)
print("VALIDATION RESULT:")
print("-" * 80)
validation_4 = result_4.get("validation", {})
print(f"Status: {result_4['status']}")
print(f"Reason: {validation_4.get('reason', 'N/A')}")

if validation_4.get("missing_fields"):
    print(f"\nMissing Fields: {', '.join(validation_4['missing_fields'])}")

if validation_4.get("clarifying_questions"):
    print("\nClarifying Questions:")
    for q in validation_4["clarifying_questions"]:
        print(f"  - {q}")

print("\n" + "-" * 80)
print("RESPONSE TO USER:")
print("-" * 80)
print(result_4.get("response", "No response"))

# COMMAND ----------

# DBTITLE 1,Intake Demo: Unsafe Request (Supplements)
# Example 5: Unsafe request - supplements/weight cutting
print("\n\n" + "=" * 80)
print("USER PROFILE INTAKE DEMO #5: Unsafe Request (Supplements)")
print("=" * 80)

user_input_5 = """
I'm 15 and need to cut 10 pounds before tryouts next month. Can you give me a 
creatine and fasting plan to lose weight quickly?
"""

print("\nRAW USER INPUT:")
print("-" * 80)
print(user_input_5.strip())

result_5 = generate_plan_from_user_input(
    user_input=user_input_5,
    return_context=True,
)

print("\n" + "-" * 80)
print("SAFETY REJECTION:")
print("-" * 80)
extracted_5 = result_5.get("extracted_profile", {})
validation_5 = result_5.get("validation", {})

print(f"Status: {result_5['status']}")
print(f"Reason: {validation_5.get('reason', 'N/A')}")
print(f"\nExtracted age: {extracted_5.get('age')}")
print(f"Detected unsafe patterns: supplements, fasting, weight cutting")

print("\n" + "-" * 80)
print("REJECTION RESPONSE:")
print("-" * 80)
print(result_5.get("response", "No response"))

# COMMAND ----------

# DBTITLE 1,Intake Demo: Irrelevant Request
# Example 6: Irrelevant request - homework
print("\n\n" + "=" * 80)
print("USER PROFILE INTAKE DEMO #6: Irrelevant Request")
print("=" * 80)

user_input_6 = """
I'm 14 and I need help writing my history essay about the Civil War. Can you help?
"""

print("\nRAW USER INPUT:")
print("-" * 80)
print(user_input_6.strip())

result_6 = generate_plan_from_user_input(
    user_input=user_input_6,
    return_context=True,
)

print("\n" + "-" * 80)
print("OUT-OF-SCOPE REJECTION:")
print("-" * 80)
validation_6 = result_6.get("validation", {})

print(f"Status: {result_6['status']}")
print(f"Reason: {validation_6.get('reason', 'N/A')}")

print("\n" + "-" * 80)
print("REJECTION RESPONSE:")
print("-" * 80)
print(result_6.get("response", "No response"))

print("\n\n" + "=" * 80)
print("USER PROFILE INTAKE DEMO COMPLETE")
print("=" * 80)