"""Run agent-eval-orchestrator evaluation pipeline against FinSight and save baseline output."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

# Ensure FinSight and agent-eval-orchestrator roots are in sys.path
FINSIGHT_ROOT = Path(__file__).resolve().parents[1]
EVAL_ORCHESTRATOR_ROOT = FINSIGHT_ROOT.parent / "agent-eval-orchestrator"

if str(FINSIGHT_ROOT) not in sys.path:
    sys.path.insert(0, str(FINSIGHT_ROOT))
if str(EVAL_ORCHESTRATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ORCHESTRATOR_ROOT))

from agent_eval_orchestrator.orchestrator.pipeline import create_eval_pipeline
from agent_eval_orchestrator.storage import BaselineStore
from agent_eval_orchestrator.types import EvalScenario
from finsight.eval.target_agent import FinSightTargetAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("run_eval_baseline")


async def run_evaluation_cycle() -> dict:
    """Execute one full 3x noise-tolerant evaluation cycle and save baseline."""
    scenarios_path = FINSIGHT_ROOT / "finsight" / "eval" / "eval_scenarios.json"
    baseline_path = FINSIGHT_ROOT / "finsight" / "eval" / "eval_baseline.json"
    out_verdict_path = FINSIGHT_ROOT / "finsight" / "eval" / "verdict_report.txt"

    logger.info("Loading scenarios from %s", scenarios_path)
    with open(scenarios_path, "r", encoding="utf-8") as f:
        scenarios_raw = json.load(f)

    scenarios = [EvalScenario.model_validate(item) for item in scenarios_raw]
    logger.info("Loaded %d scenarios for evaluation.", len(scenarios))

    target_agent = FinSightTargetAgent(version="finsight-langgraph-v1.0")

    logger.info("Creating agent-eval-orchestrator LangGraph evaluation graph...")
    eval_pipeline = create_eval_pipeline()

    initial_state = {
        "scenarios": scenarios,
        "target_agent": target_agent,
        "baseline_results": [],
    }

    logger.info("Executing evaluation pipeline (3x noise tolerance runs per scenario)...")
    final_state = await eval_pipeline.ainvoke(initial_state)

    graded_results = final_state.get("graded_results", [])
    verdict_report = final_state.get("final_verdict", "")

    # Save to baseline store
    store = BaselineStore(baseline_path)
    store.promote(graded_results)

    # Save verdict text
    with open(out_verdict_path, "w", encoding="utf-8") as f:
        f.write(verdict_report)

    print("\n" + verdict_report + "\n")
    print(f"Baseline graded results saved to: {baseline_path}")
    print(f"Verdict report saved to:           {out_verdict_path}\n")

    return {
        "scenario_count": len(scenarios),
        "graded_results": [r.model_dump() for r in graded_results],
        "verdict_report": verdict_report,
    }


if __name__ == "__main__":
    asyncio.run(run_evaluation_cycle())
