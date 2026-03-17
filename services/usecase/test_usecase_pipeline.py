"""
tests/test_usecase_pipeline.py

"""


import pytest
import asyncio
from unittest.mock import patch, MagicMock
from typing import Dict, Any

SAMPLE_DETECTION_OUTPUT = {
    "camera_id": "cam_1",
    "detections": [
        {"class_name": "person", "confidence": 0.91, "in_roi": True,  "bbox": {"x1": 10, "y1": 20, "x2": 50, "y2": 80}},
        {"class_name": "person", "confidence": 0.85, "in_roi": True,  "bbox": {"x1": 60, "y1": 20, "x2": 100, "y2": 80}},
        {"class_name": "person", "confidence": 0.72, "in_roi": False, "bbox": {"x1": 200, "y1": 20, "x2": 240, "y2": 80}},
        {"class_name": "car",    "confidence": 0.88, "in_roi": True,  "bbox": {"x1": 300, "y1": 100, "x2": 400, "y2": 200}},
    ],
    "screenshot_path": "/screenshots/cam_1_20240310.jpg",
    "first_detection_id": 42,
    "total_detections_count": 4,
    "roi_detections_count": 3,
}


CROWD_DETECTION_OUTPUT = {
    **SAMPLE_DETECTION_OUTPUT,
    "detections": [
        {"class_name": "person", "confidence": 0.91, "in_roi": True, "bbox": {}},
        {"class_name": "person", "confidence": 0.85, "in_roi": True, "bbox": {}},
        {"class_name": "person", "confidence": 0.78, "in_roi": True, "bbox": {}},
        {"class_name": "person", "confidence": 0.65, "in_roi": True, "bbox": {}},
    ],
}
# *************************************************************************************
# Test the engine(pure logic)
# ************************************************************************************
class TestEngine:
    """
    Tests for usecase/engine.py - no rabbitmq, no DB, no FastAPI
    """

    def test_build_slim_payload_strips_unnecessary_fields(self):
        from usecase.engine import build_slim_payload

        slim = build_slim_payload(SAMPLE_DETECTION_OUTPUT)

        assert "detections" in slim
        assert len(slim['detections']) == 4
        assert slim['screenshot_path'] == SAMPLE_DETECTION_OUTPUT['screenshot_path']
        assert slim['first_detection_id'] == 42
# each slim detection should only have essential fields

        first = slim['detections'][0]
        assert 'class_name' in first
        assert 'confidence' in first
        assert 'in_roi' in first

        # should not contain raw tensor data or processing metadata

        assert 'raw_tensor' not in first
   
# ************************************************************************************************


    def test_evaluate_single_person_in_roi(self):
        from usecase.engine import evaluate_single_usecase, build_slim_payload

        slim = build_slim_payload(SAMPLE_DETECTION_OUTPUT)
        result = evaluate_single_usecase('person_in_roi', slim,'cam_1')
        
        assert result.triggered is True
        assert result.matched_count == 2
        assert result.usecase_id == 'person_in_roi'


# ********************************************************************************************************


    def test_evaluate_crowd_triggers_at_threshold(self):
        from usecase.engine import evaluate_single_usecase, build_slim_payload

        slim = build_slim_payload(CROWD_DETECTION_OUTPUT)
        result = evaluate_single_usecase('crowd_in_roi', slim, 'cam_1')

        assert result.triggered is True
        assert result.matched_count == 4

# ************************************************************************************************************\

    def test_evaluate_crowd_does_not_trigger_below_threshold(self):
        from usecase.engine import evaluate_single_usecase, build_slim_payload

        two_persons ={
                **SAMPLE_DETECTION_OUTPUT,
                "detections": [
                    {"class_name": "person", "confidence": 0.9, "in_roi": True, "bbox": {}},
                    {"class_name": "person", "confidence": 0.8, "in_roi": True, "bbox": {}},
                ],
            }
        
        slim = build_slim_payload(two_persons)
        result = evaluate_single_usecase('crowd_in_roi', slim,'cam_1')

        assert result.triggered is False

    # ***************************************************************

    def test_evaluate_restricted_zone_triggers_on_vehicle(self):
        from usecase.engine import evaluate_single_usecase, build_slim_payload

        slim = build_slim_payload(SAMPLE_DETECTION_OUTPUT)
        result = evaluate_single_usecase('restricted_zone_breach', slim, 'cam_1')

        assert result.triggered is True
        assert result.matched_count == 1

    # ***********************************************************


    def test_unknown_usecase_returns_safe_default(self):
        from usecase.engine import evaluate_single_usecase, build_slim_payload

        slim = build_slim_payload(SAMPLE_DETECTION_OUTPUT)
        result = evaluate_single_usecase('nonexistent_usecase',slim, 'cam_1')

        # should not raise - returns safe default
        assert result.triggered is False
        assert result.matched_count == 0
        assert result.usecase_id == 'nonexistent_usecase'

    # *******************************************************************

    def test_matched_objects_are_slim(self):
        """
        matched objects should only contains slim fields, not full dicts
        """
        from usecase.engine import evaluate_single_usecase, build_slim_payload

        slim = build_slim_payload(SAMPLE_DETECTION_OUTPUT)
        result = evaluate_single_usecase('person_in_roi', slim,'cam_1')

        for obj in result.matched_objects:
            assert 'class_name' in obj
            assert 'in_roi' in obj
            assert 'confidence' in obj

        # *****************************************************************

    def test_evaluate_all_usecases_returns_one_result_per_usecase(self):
        from usecase.engine import evaluate_all_usecases

        usecases = ['person_in_roi', 'crowd_in_roi', 'restricted_zone_breach']
        results = evaluate_all_usecases('cam_1', SAMPLE_DETECTION_OUTPUT, usecases)
        print(f"****************Results: {(results)}*********")

        assert len(results) == 3
        usecases_ids = {r.usecase_id for r in results}
        assert usecases_ids == set(usecases)

    # ******************************************************************************
    # Test the rules(individually (Unit Tests))
    # ************************************************************************

class TestRules:
    """
    Tests for individual rule classes - pure unit tests
    """

    def test_person_in_roi_rule_directly(self):
        from usecase.rules.person_in_roi import PersonInROIRule

        rule = PersonInROIRule('person_in_roi')
        result = rule.evaluate({"detections": [
            {"class_name": "person", "in_roi": True, "confidence": 0.9},
        ]})
        assert result['triggered'] is True
        assert len(result['matched_objects']) == 1
# *******************************************************************************************
    def test_base_helper_get_in_roi_by_class(self):
        from usecase.rules.person_in_roi import PersonInROIRule

        rule = PersonInROIRule('test')
        detections = {
            "detections": [
                {"class_name": "person", "in_roi": True,  "confidence": 0.9},
                {"class_name": "car",    "in_roi": True,  "confidence": 0.8},
                {"class_name": "person", "in_roi": False, "confidence": 0.7},
            ]
        }
        person_in_roi = rule.get_in_roi_by_class(detections,['person'])
        assert len(person_in_roi) == 1 #only the one with in_roi
# ****************************************************************************************************

    def test_auto_discovery_finds_all_rules(self):
        from usecase.rules import USECASE_REGISTRY, list_usecase
        
        #All three existing rules should be discovered automatically 
        assert  'person_in_roi' in USECASE_REGISTRY
        assert 'crowd_in_roi' in USECASE_REGISTRY
        assert 'restricted_zone_breach' in USECASE_REGISTRY

        print(f'Discovered Usecase: {list_usecase()}')


#****************************************************************************************************
# Test the queue layer (mock celery)
# ***************************************************************************************************************

class TestQueueLayer:
    """
    Tests for workers/queue.ppy - celery is mocked so no rabbitMQ needed.

    Unit tests should not require external services.
    we mock the celery task's .delay() method to retuurn a fake AsyncResult

    whose .get() returns a pre-built result dict. 
    This lets us test the queue coordination logic without infrastructure
    """
    
    @pytest.mark.asyncio

    async def test_submit_and_collect_results(self):
        from workers.queue import submit_usecase_tasks, await_usecase_results
        from usecase.schemas import UsecaseResult

        fake_result_dict = {
            'usecase_id': 'person_in_roi',
            'triggered': True,
            'matched_count': 2,
            'matched_objects': [],
            'detection_id': 42,
            'screenshot_path': '/screenshot/test.jpg',
        }

        # Mock the celery tasks .delay() to return a fake AsyncResult

        mock_async_result = MagicMock()
        mock_async_result.id = 'fake-task-id-123'
        mock_async_result.get.return_value = fake_result_dict

        with patch("workers.tasks.evaluate_usecase_task.delay", return_value=mock_async_result):
            handles = submit_usecase_tasks('cam_1', SAMPLE_DETECTION_OUTPUT, ['person_in_roi'])
            assert 'person_in_roi' in handles

            results = await await_usecase_results(handles, 'cam_1')

            assert len(results) == 1
            assert results[0].triggered is True
            assert results[0].matched_count == 2
# *******************************************************************************
    @pytest.mark.asyncio
    async def test_failed_task_return_safe_default(self):
        """
        If a task raises, we get triggered=False, not an exception
        """
        from workers.queue import submit_usecase_tasks, await_usecase_results

        mock_async_result = MagicMock()
        mock_async_result.id = 'fake-task-id-456'
        mock_async_result.get.side_effect = Exception("Worker crashed")

        with patch("workers.tasks.evaluate_usecase_task.delay", return_value=mock_async_result):
            handles = submit_usecase_tasks('cam_1', SAMPLE_DETECTION_OUTPUT, ['person_in_roi'])
            results = await await_usecase_results(handles, 'cam_1')

            assert len(results) == 1
            assert results[0].triggered is False
            assert results[0].matched_count == 0

# ************************************************************
# Test the service layer (routing logic)
# ******************************************************************

class TestService:
    """
    Test for usecase/service.py - routing between direct and queue paths
    """

    @pytest.mark.asyncio
    async def test_direct_path_when_queue_disabled(self):
        import usecase.service as svc

        with patch.object(svc, 'USE_WORKER_QUEUE', False):
            response = await svc.evaluate_usecases_service(
                camera_id = 'cam_1',
                detection_output = SAMPLE_DETECTION_OUTPUT,
                usecases = ['person_in_roi'],
            )
        
        assert response.camera_id == 'cam_1'
        assert len(response.results) == 1
        assert response.results[0].usecase_id == 'person_in_roi'
# ************************************************************
    @pytest.mark.asyncio
    async def test_queue_path_when_queue_enabled(self):
        import usecase.service as svc
        from usecase.schemas import UsecaseResult

        fake_results = [
            UsecaseResult(
                usecase_id='person_in_roi',
                triggered=True,
                matched_count=1,
                matched_objects=[],
                detection_id=42,
                screenshot_path=None,   
            )
        ]
    
        with patch.object(svc, "USE_WORKER_QUEUE", True):
            with patch.object(svc,'_queue_path', return_value=fake_results) as mock_queue:
                response = await svc.evaluate_usecases_service(
                    camera_id='cam_1',
                    detection_output = SAMPLE_DETECTION_OUTPUT,
                    usecases=['person_in_roi'],
                )
                mock_queue.assert_called_once()

        assert response.results[0].triggered is True




    









