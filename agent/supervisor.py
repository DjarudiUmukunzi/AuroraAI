"""
supervisor.py

Supervisor Agent: orchestrates Research, Forecasting, and Reasoning agents
as a LangGraph graph. This is the Phase 3 "Core AI-103 Demonstration" -
a genuine multi-agent workflow rather than a flat sequential script.

The graph:
    START -> research -> forecast -> reason -> guardrail -> END

Each node updates a shared state object; the Supervisor's job is the graph
structure itself (routing, state management) plus a lightweight guardrail
check before returning the final answer.

Phase 4 addition: every run is traced via OpenTelemetry to Azure
Application Insights (observability piece of MLOps/GenAIOps). If
APPLICATIONINSIGHTS_CONNECTION_STRING isn't set, tracing is silently
skipped so the pipeline still runs fine without it.

Run directly for a quick test:
    python supervisor.py
"""

import os
from typing import TypedDict

from dotenv import load_dotenv

from forecasting_agent import forecast_next_kp
from reasoning_agent import reason
from research_agent import research

load_dotenv()

_TRACING_ENABLED = bool(os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"))
if _TRACING_ENABLED:
    from azure.monitor.opentelemetry import configure_azure_monitor

    configure_azure_monitor(connection_string=os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"])

from opentelemetry import trace

_tracer = trace.get_tracer(__name__)


class AuroraState(TypedDict, total=False):
    query: str
    research_results: list[dict]
    predicted_kp: float
    explanation: str
    guardrail_notes: list[str]


def research_node(state: AuroraState) -> AuroraState:
    with _tracer.start_as_current_span("research_node") as span:
        query = state.get("query", "geomagnetic storm aurora forecast")
        span.set_attribute("query", query)
        results = research(query)
        span.set_attribute("bulletins_retrieved", len(results))
        return {"research_results": results}


def forecast_node(state: AuroraState) -> AuroraState:
    with _tracer.start_as_current_span("forecast_node") as span:
        predicted = forecast_next_kp()
        span.set_attribute("predicted_kp", predicted)
        return {"predicted_kp": predicted}


def reasoning_node(state: AuroraState) -> AuroraState:
    with _tracer.start_as_current_span("reasoning_node") as span:
        explanation = reason(state["predicted_kp"], state["research_results"])
        span.set_attribute("explanation_length", len(explanation))
        return {"explanation": explanation}


def guardrail_node(state: AuroraState) -> AuroraState:
    """Lightweight responsible-AI check: the Supervisor's guardrail duty
    from the project plan. Flags (but doesn't block) explanations that
    read as overconfident predictions of a specific future event, since
    space weather forecasting is inherently probabilistic."""
    with _tracer.start_as_current_span("guardrail_node") as span:
        notes = []
        explanation = state.get("explanation", "")
        overconfident_phrases = ["will definitely", "guaranteed", "certainly will happen"]
        if any(p in explanation.lower() for p in overconfident_phrases):
            notes.append("Explanation contains overconfident language for a probabilistic forecast.")
        if state.get("predicted_kp", 0) < 0 or state.get("predicted_kp", 0) > 9:
            notes.append(f"Predicted Kp {state.get('predicted_kp')} is outside the valid 0-9 range.")
        span.set_attribute("guardrail_flags_count", len(notes))
        if notes:
            span.set_attribute("guardrail_flags", str(notes))
        return {"guardrail_notes": notes}


def build_graph():
    from langgraph.graph import END, StateGraph

    graph = StateGraph(AuroraState)
    graph.add_node("research", research_node)
    graph.add_node("forecast", forecast_node)
    graph.add_node("reason", reasoning_node)
    graph.add_node("guardrail", guardrail_node)

    graph.set_entry_point("research")
    graph.add_edge("research", "forecast")
    graph.add_edge("forecast", "reason")
    graph.add_edge("reason", "guardrail")
    graph.add_edge("guardrail", END)

    return graph.compile()


def run(query: str = "geomagnetic storm aurora forecast") -> AuroraState:
    app = build_graph()
    with _tracer.start_as_current_span("supervisor_run") as span:
        span.set_attribute("query", query)
        result = app.invoke({"query": query})
        span.set_attribute("predicted_kp", result.get("predicted_kp", -1))
        span.set_attribute("guardrail_flags_count", len(result.get("guardrail_notes", [])))
        return result


if __name__ == "__main__":
    print(f"Tracing to Application Insights: {'enabled' if _TRACING_ENABLED else 'disabled (no connection string set)'}")
    result = run()
    print(f"\nPredicted Kp: {result['predicted_kp']:.2f}")
    print(f"\nExplanation:\n{result['explanation']}")
    if result["guardrail_notes"]:
        print(f"\n[Guardrail flags]: {result['guardrail_notes']}")
    else:
        print("\n[Guardrail]: no issues flagged.")