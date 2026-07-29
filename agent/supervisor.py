"""
supervisor.py

Supervisor Agent: orchestrates Research, Forecasting, and Reasoning agents
as a LangGraph graph. This is the Phase 3
a genuine multi-agent workflow rather than a flat sequential script.

The graph:
    START -> research -> forecast -> reason -> END

Each node updates a shared state object; the Supervisor's job is the graph
structure itself (routing, state management) plus a lightweight guardrail
check before returning the final answer.

Run directly for a quick test:
    python supervisor.py
"""

from typing import TypedDict

from forecasting_agent import forecast_next_kp
from reasoning_agent import reason
from research_agent import research


class AuroraState(TypedDict, total=False):
    query: str
    research_results: list[dict]
    predicted_kp: float
    explanation: str
    guardrail_notes: list[str]


def research_node(state: AuroraState) -> AuroraState:
    query = state.get("query", "geomagnetic storm aurora forecast")
    results = research(query)
    return {"research_results": results}


def forecast_node(state: AuroraState) -> AuroraState:
    predicted = forecast_next_kp()
    return {"predicted_kp": predicted}


def reasoning_node(state: AuroraState) -> AuroraState:
    explanation = reason(state["predicted_kp"], state["research_results"])
    return {"explanation": explanation}


def guardrail_node(state: AuroraState) -> AuroraState:
    """Lightweight responsible-AI check: the Supervisor's guardrail duty
    from the project plan. Flags (but doesn't block) explanations that
    read as overconfident predictions of a specific future event, since
    space weather forecasting is inherently probabilistic."""
    notes = []
    explanation = state.get("explanation", "")
    overconfident_phrases = ["will definitely", "guaranteed", "certainly will happen"]
    if any(p in explanation.lower() for p in overconfident_phrases):
        notes.append("Explanation contains overconfident language for a probabilistic forecast.")
    if state.get("predicted_kp", 0) < 0 or state.get("predicted_kp", 0) > 9:
        notes.append(f"Predicted Kp {state.get('predicted_kp')} is outside the valid 0-9 range.")
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
    return app.invoke({"query": query})


if __name__ == "__main__":
    result = run()
    print(f"\nPredicted Kp: {result['predicted_kp']:.2f}")
    print(f"\nExplanation:\n{result['explanation']}")
    if result["guardrail_notes"]:
        print(f"\n[Guardrail flags]: {result['guardrail_notes']}")
    else:
        print("\n[Guardrail]: no issues flagged.")