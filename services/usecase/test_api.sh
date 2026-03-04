#!/bin/bash

# Test script for Usecase Evaluation Service
# Tests all available usecases with sample detection data

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Service URL
SERVICE_URL="${USECASE_SERVICE_URL:-http://localhost:8001}"

echo "=================================================="
echo "Usecase Evaluation Service - Test Suite"
echo "=================================================="
echo "Service URL: $SERVICE_URL"
echo ""

# Function to run test
run_test() {
    local test_name="$1"
    local endpoint="$2"
    local data="$3"
    local expected_triggered="$4"
    
    echo -e "${YELLOW}Testing: $test_name${NC}"
    
    response=$(curl -s -X POST "$SERVICE_URL$endpoint" \
        -H "Content-Type: application/json" \
        -d "$data")
    
    if [ $? -eq 0 ]; then
        # Check if response contains expected result
        if echo "$response" | grep -q "\"triggered\":$expected_triggered"; then
            echo -e "${GREEN}✓ PASS${NC}: $test_name"
            echo "  Response: $response" | head -c 200
            echo ""
        else
            echo -e "${RED}✗ FAIL${NC}: $test_name"
            echo "  Expected triggered: $expected_triggered"
            echo "  Response: $response"
        fi
    else
        echo -e "${RED}✗ FAIL${NC}: Request failed for $test_name"
    fi
    echo ""
}

# Test 1: Health check
echo "------------------------------------------------"
echo "Test 1: Health Check"
echo "------------------------------------------------"
health_response=$(curl -s "$SERVICE_URL/health")
if echo "$health_response" | grep -q "healthy"; then
    echo -e "${GREEN}✓ PASS${NC}: Service is healthy"
    echo "  Response: $health_response"
else
    echo -e "${RED}✗ FAIL${NC}: Service health check failed"
    echo "  Response: $health_response"
    exit 1
fi
echo ""

# Test 2: Person in ROI (should trigger)
echo "------------------------------------------------"
echo "Test 2: Person in ROI - Should Trigger"
echo "------------------------------------------------"
run_test "Person in ROI (triggered)" \
    "/usecase/evaluate" \
    '{
        "camera_id": "test_cam_1",
        "detection_output": {
            "camera_id": "test_cam_1",
            "detections": [
                {
                    "class_name": "person",
                    "confidence": 0.92,
                    "in_roi": true,
                    "bbox": [100, 150, 200, 350]
                }
            ],
            "screenshot_path": "/screenshots/test1.jpg",
            "first_detection_id": 1
        },
        "usecases": ["person_in_roi"]
    }' \
    "true"

# Test 3: Person NOT in ROI (should not trigger)
echo "------------------------------------------------"
echo "Test 3: Person NOT in ROI - Should Not Trigger"
echo "------------------------------------------------"
run_test "Person outside ROI (not triggered)" \
    "/usecase/evaluate" \
    '{
        "camera_id": "test_cam_2",
        "detection_output": {
            "camera_id": "test_cam_2",
            "detections": [
                {
                    "class_name": "person",
                    "confidence": 0.85,
                    "in_roi": false,
                    "bbox": [500, 500, 600, 600]
                }
            ],
            "screenshot_path": "/screenshots/test2.jpg",
            "first_detection_id": 2
        },
        "usecases": ["person_in_roi"]
    }' \
    "false"

# Test 4: Crowd in ROI (3+ persons, should trigger)
echo "------------------------------------------------"
echo "Test 4: Crowd in ROI - Should Trigger"
echo "------------------------------------------------"
run_test "Crowd in ROI (3 persons)" \
    "/usecase/evaluate" \
    '{
        "camera_id": "test_cam_3",
        "detection_output": {
            "camera_id": "test_cam_3",
            "detections": [
                {"class_name": "person", "confidence": 0.9, "in_roi": true, "bbox": [100, 100, 200, 200]},
                {"class_name": "person", "confidence": 0.85, "in_roi": true, "bbox": [300, 100, 400, 200]},
                {"class_name": "person", "confidence": 0.88, "in_roi": true, "bbox": [500, 100, 600, 200]}
            ],
            "screenshot_path": "/screenshots/test3.jpg",
            "first_detection_id": 3
        },
        "usecases": ["crowd_in_roi"]
    }' \
    "true"

# Test 5: Small group (2 persons, should not trigger)
echo "------------------------------------------------"
echo "Test 5: Small Group - Should Not Trigger"
echo "------------------------------------------------"
run_test "Small group (2 persons, not crowd)" \
    "/usecase/evaluate" \
    '{
        "camera_id": "test_cam_4",
        "detection_output": {
            "camera_id": "test_cam_4",
            "detections": [
                {"class_name": "person", "confidence": 0.9, "in_roi": true, "bbox": [100, 100, 200, 200]},
                {"class_name": "person", "confidence": 0.85, "in_roi": true, "bbox": [300, 100, 400, 200]}
            ],
            "screenshot_path": "/screenshots/test4.jpg",
            "first_detection_id": 4
        },
        "usecases": ["crowd_in_roi"]
    }' \
    "false"

# Test 6: Restricted zone breach (car in ROI, should trigger)
echo "------------------------------------------------"
echo "Test 6: Restricted Zone Breach - Should Trigger"
echo "------------------------------------------------"
run_test "Vehicle in restricted zone" \
    "/usecase/evaluate" \
    '{
        "camera_id": "test_cam_5",
        "detection_output": {
            "camera_id": "test_cam_5",
            "detections": [
                {"class_name": "car", "confidence": 0.92, "in_roi": true, "bbox": [200, 200, 400, 350]}
            ],
            "screenshot_path": "/screenshots/test5.jpg",
            "first_detection_id": 5
        },
        "usecases": ["restricted_zone_breach"]
    }' \
    "true"

# Test 7: Multiple usecases at once
echo "------------------------------------------------"
echo "Test 7: Multiple Usecases Evaluation"
echo "------------------------------------------------"
echo -e "${YELLOW}Testing: Multiple usecases (person_in_roi + restricted_zone_breach)${NC}"
response=$(curl -s -X POST "$SERVICE_URL/usecase/evaluate" \
    -H "Content-Type: application/json" \
    -d '{
        "camera_id": "test_cam_6",
        "detection_output": {
            "camera_id": "test_cam_6",
            "detections": [
                {"class_name": "person", "confidence": 0.9, "in_roi": true, "bbox": [100, 100, 200, 200]},
                {"class_name": "car", "confidence": 0.88, "in_roi": true, "bbox": [400, 200, 600, 350]}
            ],
            "screenshot_path": "/screenshots/test6.jpg",
            "first_detection_id": 6
        },
        "usecases": ["person_in_roi", "restricted_zone_breach"]
    }')

if echo "$response" | grep -q "\"usecase_id\":\"person_in_roi\"" && \
   echo "$response" | grep -q "\"usecase_id\":\"restricted_zone_breach\""; then
    echo -e "${GREEN}✓ PASS${NC}: Multiple usecases evaluated"
    echo "  Both person_in_roi and restricted_zone_breach returned"
else
    echo -e "${RED}✗ FAIL${NC}: Multiple usecases test failed"
fi
echo ""

# Test 8: Invalid usecase (should handle gracefully)
echo "------------------------------------------------"
echo "Test 8: Invalid Usecase - Error Handling"
echo "------------------------------------------------"
echo -e "${YELLOW}Testing: Invalid usecase ID${NC}"
response=$(curl -s -X POST "$SERVICE_URL/usecase/evaluate" \
    -H "Content-Type: application/json" \
    -d '{
        "camera_id": "test_cam_7",
        "detection_output": {
            "camera_id": "test_cam_7",
            "detections": [
                {"class_name": "person", "confidence": 0.9, "in_roi": true, "bbox": [100, 100, 200, 200]}
            ],
            "screenshot_path": "/screenshots/test7.jpg",
            "first_detection_id": 7
        },
        "usecases": ["invalid_usecase_name"]
    }')

# Service should return empty results or handle gracefully (not crash)
if [ ! -z "$response" ]; then
    echo -e "${GREEN}✓ PASS${NC}: Service handled invalid usecase gracefully"
else
    echo -e "${RED}✗ FAIL${NC}: Service did not respond"
fi
echo ""

# Final summary
echo "=================================================="
echo "Test Suite Completed!"
echo "=================================================="
echo ""
echo "Review the results above to ensure all tests passed."
echo ""
echo "To view service logs:"
echo "  docker-compose logs -f usecase-service"
echo ""
echo "To check database records:"
echo "  docker-compose exec postgres psql -U goec -d goec -c 'SELECT * FROM usecase_results ORDER BY timestamp DESC LIMIT 10;'"
echo ""
