"""
evaluate_agents.py

GenAIOps piece of Phase 4: runs the Supervisor's multi-agent pipeline
against a fixed set of test queries and scores each response on three
axes, the standard RAG evaluation triad:

  - Groundedness: does the explanation actually draw on the retrieved
    bulletins, rather than hallucinating unsupported claims? (LLM-as-judge)
  - Relevance: were the retrieved bulletins actually on-topic for the
    query? (embedding similarity between query and retrieved content)
  - Safety: does the explanation avoid overconfident/unsafe claims?
    (reuses the Supervisor's guardrail_node logic)

Run:
    python agent/evaluate_agents.py
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))

from research_agent import research, _get_openai_client, _embed_with_retry, EMBEDDING_DEPLOYMENT
from supervisor import run as run_supervisor
from supervisor import guardrail_node

load_dotenv()

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")

TEST_QUERIES = [
    "geomagnetic storm warning solar wind",
    "aurora visibility forecast tonight",
    "solar flare radio blackout risk",
]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def score_relevance(query: str, research_results: list[dict]) -> float:
    """Average cosine similarity between the query and each retrieved
    bulletin's embedding. Higher = retrieval actually stayed on-topic."""
    if not research_results:
        return 0.0
    client = _get_openai_client()
    embed_model = EMBEDDING_DEPLOYMENT

    query_vec = _embed_with_retry(client, [query]).data[0].embedding
    texts = [r["content"][:2000] for r in research_results]
    doc_vecs = _embed_with_retry(client, texts).data

    sims = [_cosine_sim(query_vec, d.embedding) for d in doc_vecs]
    return round(sum(sims) / len(sims), 3)


def score_groundedness(explanation: str, research_results: list[dict], predicted_kp: float) -> dict:
    """LLM-as-judge: does the explanation's claims trace back to the
    retrieved bulletins and the forecast value, rather than inventing
    unsupported specifics?"""
    client = _get_openai_client()
    bulletins_text = "\n\n".join(r["content"][:500] for r in research_results)

    judge_prompt = f"""You are an evaluator checking whether an AI-generated
explanation is grounded in its source material. Score groundedness from
1 (mostly unsupported/hallucinated) to 5 (every claim traceable to sources).

Predicted Kp index (source data): {predicted_kp:.2f}

Retrieved bulletins (source data):
{bulletins_text}

Explanation to evaluate:
{explanation}

Respond with ONLY a JSON object: {{"score": <1-5>, "reasoning": "<one sentence>"}}"""

    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "user", "content": judge_prompt}],
        max_completion_tokens=600,
        reasoning_effort="low",
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        # Strip markdown fences if the model added them despite instructions
        raw = raw.strip("`").removeprefix("json").strip()
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {"score": None, "reasoning": f"[unparsed judge output] {raw[:200]}"}


def evaluate_one(query: str) -> dict:
    result = run_supervisor(query)
    relevance = score_relevance(query, result["research_results"])
    groundedness = score_groundedness(
        result["explanation"], result["research_results"], result["predicted_kp"]
    )
    safety_flags = result["guardrail_notes"]

    return {
        "query": query,
        "predicted_kp": result["predicted_kp"],
        "relevance_score": relevance,
        "groundedness_score": groundedness.get("score"),
        "groundedness_reasoning": groundedness.get("reasoning"),
        "safety_flags": safety_flags,
        "safety_pass": len(safety_flags) == 0,
    }


def main():
    print(f"Evaluating {len(TEST_QUERIES)} test queries against the Supervisor pipeline...\n")
    results = []
    for i, q in enumerate(TEST_QUERIES):
        print(f"  -> {q}")
        results.append(evaluate_one(q))
        if i < len(TEST_QUERIES) - 1:
            time.sleep(3)  # brief pause between queries to avoid clustering embedding calls

    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)
    for r in results:
        print(f"\nQuery: {r['query']}")
        print(f"  Predicted Kp:       {r['predicted_kp']:.2f}")
        print(f"  Relevance:          {r['relevance_score']}  (cosine similarity, 0-1)")
        print(f"  Groundedness:       {r['groundedness_score']}/5  - {r['groundedness_reasoning']}")
        print(f"  Safety:             {'PASS' if r['safety_pass'] else 'FLAGGED: ' + str(r['safety_flags'])}")

    avg_relevance = sum(r["relevance_score"] for r in results) / len(results)
    grounded_scores = [r["groundedness_score"] for r in results if r["groundedness_score"] is not None]
    avg_groundedness = sum(grounded_scores) / len(grounded_scores) if grounded_scores else None
    safety_pass_rate = sum(r["safety_pass"] for r in results) / len(results)

    print("\n" + "-" * 70)
    print(f"Avg relevance:     {avg_relevance:.3f}")
    print(f"Avg groundedness:  {avg_groundedness:.2f}/5" if avg_groundedness else "Avg groundedness:  N/A")
    print(f"Safety pass rate:  {safety_pass_rate:.0%}")

    report_path = os.path.join(os.path.dirname(__file__), "..", "eval_report.json")
    with open(report_path, "w") as f:
        json.dump(
            {
                "run_at": datetime.now(timezone.utc).isoformat(),
                "results": results,
                "summary": {
                    "avg_relevance": avg_relevance,
                    "avg_groundedness": avg_groundedness,
                    "safety_pass_rate": safety_pass_rate,
                },
            },
            f,
            indent=2,
        )
    print(f"\nFull report saved -> {report_path}")


if __name__ == "__main__":
    main()