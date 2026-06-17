# Elite-Athletic---AI-Advisor
Agentic-AI project

An AI-powered youth athletic development advisor that combines fitness benchmarks, nutrition science, exercise databases, and safety guidelines to generate personalized weekly training, recovery, and nutrition plans for young athletes.

## LLM and Agent Setup

Run the notebooks/scripts in this order in Databricks:

1. `01_data_pipeline.py` to create the Bronze, Silver, and Gold tables.
2. `02_agent_definition.py` to define the tools, profile-intake flow, and two-LLM agent.
3. `03_evaluate_multi_llm.py` to compare the two LLM endpoints and save evaluation tables.

The agent now uses two LLM roles:

- `model_endpoint`: primary planner LLM that drafts the athlete plan.
- `review_model_endpoint`: second LLM that reviews the draft for safety, grounding, clarity, and human-review needs.

For evaluation, set these Databricks widgets:

- `primary_model_endpoint`: first model serving endpoint.
- `comparison_model_endpoint`: second model serving endpoint.
- `judge_model_endpoint`: optional evaluator endpoint. If unset, the primary endpoint is used.
- `allow_mock_llm`: set to `true` for a local classroom demo without live model endpoints.

Example calls:

```python
generate_weekly_plan(
    athlete_id="A003",
    request="Build a conditioning week without aggravating knee recovery.",
    model_endpoint=PRIMARY_MODEL_ENDPOINT,
    review_model_endpoint=COMPARISON_MODEL_ENDPOINT,
    return_context=True,
)

generate_plan_from_user_input(
    user_input="I'm 16, play shooting guard, want better explosiveness, have knee soreness, and can train 4 days a week.",
    model_endpoint=PRIMARY_MODEL_ENDPOINT,
    review_model_endpoint=COMPARISON_MODEL_ENDPOINT,
    return_context=True,
)
```
