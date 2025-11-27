"""
MADL Engine - Core Integration Layer
This file connects your FastAPI layer to the optimized semantic search engine.

Responsibilities:
 - Expose a global `optimized_pipeline`
 - Expose `analyze_test_steps_for_reusability()` which:
      → groups BDD steps
      → builds combined search queries
      → performs semantic vector search in OpenSearch
      → returns matches formatted for UI
"""

from typing import List, Dict, Any, Optional
import logging

# Import the new OpenSearch-powered processor and pipeline
from madl_engine.optimized_search import gherkin_processor, _global_pipeline

log = logging.getLogger("madl_core")
log.setLevel(logging.DEBUG)


# -----------------------------------------------------------
# analyze_test_steps_for_reusability
# -----------------------------------------------------------
def analyze_test_steps_for_reusability(steps: List[str], context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Full reusable-method analysis pipeline.

    Input:
        steps : List[str]  → BDD test steps only
        context: Optional[Dict] → extra runtime context (e.g. current_page, previous_steps)

    Output:
        [
            {
                "step_label": "2_4",
                "step_details": "When ...\nAnd ...\nAnd ...",
                "query": "login with username and password",
                "matches": [
                    {
                        "method_name": "...",
                        "class_name": "...",
                        "signature": "...",
                        "method_code": "...",
                        "score": 92.2
                    }
                ]
            }
        ]
    """

    if not steps:
        log.warning("analyze_test_steps_for_reusability: Received empty steps list.")
        return []

    # Clean steps (remove blanks)
    steps = [s.strip() for s in steps if s and s.strip()]

    if not steps:
        log.warning("analyze_test_steps_for_reusability: Steps empty after cleaning.")
        return []

    try:
        # Delegates EVERYTHING to optimized_search → GherkinSearchProcessor
        # forward optional context for context-aware scoring
        result = gherkin_processor.analyze(steps, context=context)
        return result

    except Exception as e:
        log.exception("Error in analyze_test_steps_for_reusability: %s", e)
        return []


# Export the pipeline object so other modules can access it directly if needed
optimized_pipeline = _global_pipeline
