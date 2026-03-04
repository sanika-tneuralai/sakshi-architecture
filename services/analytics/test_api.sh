#!/bin/bash

# Analytics Service API Test Script
# Tests all analytics endpoints

BASE_URL="http://localhost:8003"
echo "Testing Analytics Service at $BASE_URL"
echo "==========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test counter
PASSED=0
FAILED=0

# Helper function for tests
test_endpoint() {
    local name=$1
    local method=$2
    local endpoint=$3
    local data=$4
    
    echo "Testing: $name"
    
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" "$BASE_URL$endpoint")
    elif [ "$method" = "POST" ]; then
        response=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL$endpoint" \
            -H "Content-Type: application/json" \
            -d "$data")
    fi
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" -eq 200 ]; then
        echo -e "${GREEN}✓ PASSED${NC} (HTTP $http_code)"
        echo "Response: $body" | head -c 200
        echo ""
        ((PASSED++))
    else
        echo -e "${RED}✗ FAILED${NC} (HTTP $http_code)"
        echo "Response: $body"
        ((FAILED++))
    fi
    echo ""
}

# 1. Health Check
test_endpoint "Health Check" "GET" "/health"

# 2. Root Endpoint
test_endpoint "Root Endpoint" "GET" "/"

# 3. Get Daily Analytics - All
test_endpoint "Get All Daily Analytics" "GET" "/analytics/daily"

# 4. Get Daily Analytics - With Camera Filter
test_endpoint "Get Daily Analytics (Camera Filter)" "GET" "/analytics/daily?camera_id=cam_001"

# 5. Get Daily Analytics - With Date Range
test_endpoint "Get Daily Analytics (Date Range)" "GET" "/analytics/daily?start_date=2024-01-01&end_date=2024-12-31"

# 6. Get Alert Analytics - All
test_endpoint "Get All Alert Analytics" "GET" "/analytics/alerts"

# 7. Get Alert Analytics - With Camera Filter
test_endpoint "Get Alert Analytics (Camera Filter)" "GET" "/analytics/alerts?camera_id=cam_001"

# 8. Get Alert Analytics - With Date Range
test_endpoint "Get Alert Analytics (Date Range)" "GET" "/analytics/alerts?start_date=2024-01-01&end_date=2024-12-31"

# 9. Get Detection Analytics - All
test_endpoint "Get All Detection Analytics" "GET" "/analytics/detections"

# 10. Get Detection Analytics - With Camera Filter
test_endpoint "Get Detection Analytics (Camera Filter)" "GET" "/analytics/detections?camera_id=cam_001"

# 11. Get Detection Analytics - With Date Range
test_endpoint "Get Detection Analytics (Date Range)" "GET" "/analytics/detections?start_date=2024-01-01&end_date=2024-12-31"

# Summary
echo "==========================================="
echo "Test Summary"
echo "==========================================="
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo "Total: $((PASSED + FAILED))"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
fi
