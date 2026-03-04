#!/bin/bash

# Test script for Alert Service
# Tests alert processing and delivery

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Service URL
SERVICE_URL="${ALERT_SERVICE_URL:-http://localhost:8002}"

echo "=================================================="
echo "Alert Service - Test Suite"
echo "=================================================="
echo "Service URL: $SERVICE_URL"
echo ""

# Function to run test
run_test() {
    local test_name="$1"
    local endpoint="$2"
    local data="$3"
    local expected_pattern="$4"
    
    echo -e "${YELLOW}Testing: $test_name${NC}"
    
    response=$(curl -s -X POST "$SERVICE_URL$endpoint" \
        -H "Content-Type: application/json" \
        -d "$data")
    
    if [ $? -eq 0 ]; then
        # Check if response contains expected result
        if echo "$response" | grep -q "$expected_pattern"; then
            echo -e "${GREEN}✓ PASS${NC}: $test_name"
            echo "  Response: $response" | head -c 200
            echo ""
        else
            echo -e "${RED}✗ FAIL${NC}: $test_name"
            echo "  Expected pattern: $expected_pattern"
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

# Test 2: Root endpoint
echo "------------------------------------------------"
echo "Test 2: Root Endpoint"
echo "------------------------------------------------"
root_response=$(curl -s "$SERVICE_URL/")
if echo "$root_response" | grep -q "Alert Service"; then
    echo -e "${GREEN}✓ PASS${NC}: Root endpoint working"
    echo "  Response: $root_response"
else
    echo -e "${RED}✗ FAIL${NC}: Root endpoint failed"
    echo "  Response: $root_response"
fi
echo ""

# Test 3: Person in ROI alert
echo "------------------------------------------------"
echo "Test 3: Person in ROI - Alert Should Be Sent"
echo "------------------------------------------------"
run_test "Person in ROI alert" \
    "/alert/send" \
    '{
        "camera_id": "test_cam_1",
        "usecase_results": [{
            "usecase_id": "person_in_roi",
            "triggered": true,
            "matched_count": 1,
            "matched_objects": ["person"],
            "detection_id": 1,
            "screenshot_path": "/screenshots/test1.jpg"
        }]
    }' \
    '"total_alerts_sent":1'

# Test 4: Crowd in ROI - Below threshold
echo "------------------------------------------------"
echo "Test 4: Crowd in ROI - Below Threshold (2 persons)"
echo "------------------------------------------------"
run_test "Crowd below threshold" \
    "/alert/send" \
    '{
        "camera_id": "test_cam_2",
        "usecase_results": [{
            "usecase_id": "crowd_in_roi",
            "triggered": true,
            "matched_count": 2,
            "matched_objects": ["person", "person"],
            "detection_id": 2,
            "screenshot_path": "/screenshots/test2.jpg"
        }]
    }' \
    '"total_alerts_sent":0'

# Test 5: Crowd in ROI - Above threshold
echo "------------------------------------------------"
echo "Test 5: Crowd in ROI - Above Threshold (5 persons)"
echo "------------------------------------------------"
run_test "Crowd above threshold" \
    "/alert/send" \
    '{
        "camera_id": "test_cam_3",
        "usecase_results": [{
            "usecase_id": "crowd_in_roi",
            "triggered": true,
            "matched_count": 5,
            "matched_objects": ["person", "person", "person", "person", "person"],
            "detection_id": 3,
            "screenshot_path": "/screenshots/test3.jpg"
        }]
    }' \
    '"total_alerts_sent":1'

# Test 6: Restricted zone breach
echo "------------------------------------------------"
echo "Test 6: Restricted Zone Breach - Car Detected"
echo "------------------------------------------------"
run_test "Restricted zone breach" \
    "/alert/send" \
    '{
        "camera_id": "test_cam_4",
        "usecase_results": [{
            "usecase_id": "restricted_zone_breach",
            "triggered": true,
            "matched_count": 1,
            "matched_objects": ["car"],
            "detection_id": 4,
            "screenshot_path": "/screenshots/test4.jpg"
        }]
    }' \
    '"total_alerts_sent":1'

# Test 7: Multiple alerts in one request
echo "------------------------------------------------"
echo "Test 7: Multiple Alerts in Single Request"
echo "------------------------------------------------"
run_test "Multiple alerts" \
    "/alert/send" \
    '{
        "camera_id": "test_cam_5",
        "usecase_results": [
            {
                "usecase_id": "person_in_roi",
                "triggered": true,
                "matched_count": 1,
                "matched_objects": ["person"],
                "detection_id": 5,
                "screenshot_path": "/screenshots/test5a.jpg"
            },
            {
                "usecase_id": "restricted_zone_breach",
                "triggered": true,
                "matched_count": 1,
                "matched_objects": ["truck"],
                "detection_id": 6,
                "screenshot_path": "/screenshots/test5b.jpg"
            }
        ]
    }' \
    '"total_alerts_sent":2'

# Test 8: No alerts when not triggered
echo "------------------------------------------------"
echo "Test 8: Not Triggered - No Alerts"
echo "------------------------------------------------"
run_test "Not triggered" \
    "/alert/send" \
    '{
        "camera_id": "test_cam_6",
        "usecase_results": [{
            "usecase_id": "person_in_roi",
            "triggered": false,
            "matched_count": 0,
            "matched_objects": [],
            "detection_id": 7,
            "screenshot_path": "/screenshots/test6.jpg"
        }]
    }' \
    '"total_alerts_sent":0'

# Test 9: Legacy single alert endpoint
echo "------------------------------------------------"
echo "Test 9: Legacy Single Alert Endpoint"
echo "------------------------------------------------"
run_test "Legacy single alert" \
    "/alert/send-single" \
    '{
        "camera_id": "test_cam_7",
        "usecase_id": "person_in_roi",
        "alert_required": true,
        "alert_type": "person_detected",
        "alert_objects": [{"class": "person", "confidence": 0.95}],
        "alert_count": 1
    }' \
    '"alert_sent":true'

# Test 10: Get alert list
echo "------------------------------------------------"
echo "Test 10: Get Alert List"
echo "------------------------------------------------"
list_response=$(curl -s "$SERVICE_URL/alert/list?limit=5")
if echo "$list_response" | grep -q "alerts"; then
    echo -e "${GREEN}✓ PASS${NC}: Alert list retrieved"
    echo "  Response: $list_response" | head -c 300
    echo ""
else
    echo -e "${RED}✗ FAIL${NC}: Failed to retrieve alert list"
    echo "  Response: $list_response"
fi
echo ""

# Test 11: Get alert list filtered by camera
echo "------------------------------------------------"
echo "Test 11: Get Alert List Filtered by Camera"
echo "------------------------------------------------"
filtered_response=$(curl -s "$SERVICE_URL/alert/list?camera_id=test_cam_1&limit=10")
if echo "$filtered_response" | grep -q "alerts"; then
    echo -e "${GREEN}✓ PASS${NC}: Filtered alert list retrieved"
    echo "  Response: $filtered_response" | head -c 300
    echo ""
else
    echo -e "${RED}✗ FAIL${NC}: Failed to retrieve filtered alert list"
    echo "  Response: $filtered_response"
fi
echo ""

# Summary
echo "=================================================="
echo "Test Suite Complete"
echo "=================================================="
echo ""
echo "Note: Some tests may show 'FAIL' if database is not set up."
echo "Run 'docker-compose up -d' to start the service with database."
echo ""
