"""
Usecase evaluation service - orchestrates multiple usecase rules.
"""
from typing import Dict, Any, List
from usecase.rules import get_usecase_rule
from usecase.schemas import UsecaseResult


def evaluate_usecases(camera_id: str, detection_output: Dict[str, Any], usecases: List[str]) -> Dict[str, Any]:
    """
    Evaluate multiple usecases against a single detection output.
    
    This is the orchestrator that:
    - Takes ONE detection output
    - Runs MULTIPLE usecase rules on it
    - Returns aggregated results
    
    Args:
        camera_id: Camera identifier
        detection_output: Detection API response (computed ONCE)
        usecases: List of usecase IDs to evaluate
        
    Returns:
        Dictionary containing:
            - camera_id: Camera identifier
            - results: List of UsecaseResult objects
    """
    print(f"\n[ORCHESTRATOR] ============================================================")
    print(f"[ORCHESTRATOR] USECASE EVALUATION ORCHESTRATOR")
    print(f"[ORCHESTRATOR] ============================================================")
    print(f"[ORCHESTRATOR] Camera ID: {camera_id}")
    print(f"[ORCHESTRATOR] Number of usecases to evaluate: {len(usecases)}")
    print(f"[ORCHESTRATOR] Usecases: {', '.join(usecases)}")
    
    detections = detection_output.get("detections", [])
    screenshot_path = detection_output.get("screenshot_path")
    first_detection_id = detection_output.get("first_detection_id")
    print(f"[ORCHESTRATOR] Detection output contains: {len(detections)} detections")
    print(f"[ORCHESTRATOR] Screenshot path: {screenshot_path}")
    print(f"[ORCHESTRATOR] First detection ID: {first_detection_id}")
    print(f"[ORCHESTRATOR] Detection will be evaluated ONCE for ALL usecases")
    print(f"[ORCHESTRATOR] ============================================================\n")
    
    results = []
    
    for idx, usecase_id in enumerate(usecases):
        print(f"[ORCHESTRATOR] --- Evaluating Usecase {idx+1}/{len(usecases)}: '{usecase_id}' ---")
        
        try:
            # Get the rule instance for this usecase
            rule = get_usecase_rule(usecase_id)
            print(f"[ORCHESTRATOR] Rule loaded: {rule.__class__.__name__}")
            
            # Evaluate the rule against detection output
            print(f"[ORCHESTRATOR] Calling rule.evaluate() for '{usecase_id}'...")
            evaluation_result = rule.evaluate(detection_output)
            
            # Build result object
            result = UsecaseResult(
                usecase_id=usecase_id,
                triggered=evaluation_result["triggered"],
                matched_count=len(evaluation_result["matched_objects"]),
                matched_objects=evaluation_result["matched_objects"],
                detection_id=first_detection_id,
                screenshot_path=screenshot_path
            )
            
            print(f"[ORCHESTRATOR] Usecase '{usecase_id}' result:")
            print(f"[ORCHESTRATOR]   - Triggered: {result.triggered}")
            print(f"[ORCHESTRATOR]   - Matched count: {result.matched_count}")
            
            results.append(result)
            
            # Persist to database
            try:
                from shared.database.persistence import persist_usecase_result
                persist_usecase_result(
                    camera_id=camera_id,
                    usecase_name=usecase_id,
                    triggered=result.triggered,
                    detection_id=first_detection_id
                )
            except Exception as e:
                print(f"[ORCHESTRATOR] DB Error: {str(e)}")
            
        except ValueError as e:
            print(f"[ORCHESTRATOR] ERROR: {str(e)}")
            # Skip unknown usecase, don't fail entire evaluation
            continue
        except Exception as e:
            print(f"[ORCHESTRATOR] ERROR: Usecase '{usecase_id}' evaluation failed: {str(e)}")
            # Skip failed usecase, don't fail entire evaluation
            continue
    
    print(f"\n[ORCHESTRATOR] ============================================================")
    print(f"[ORCHESTRATOR] ALL USECASES EVALUATED")
    print(f"[ORCHESTRATOR] Total results: {len(results)}/{len(usecases)}")
    
    triggered_count = sum(1 for r in results if r.triggered)
    print(f"[ORCHESTRATOR] Triggered usecases: {triggered_count}")
    
    for result in results:
        status = "✓ TRIGGERED" if result.triggered else "✗ Not triggered"
        print(f"[ORCHESTRATOR]   - {result.usecase_id}: {status} ({result.matched_count} matches)")
    
    print(f"[ORCHESTRATOR] ============================================================\n")
    
    return {
        "camera_id": camera_id,
        "results": results
    }
