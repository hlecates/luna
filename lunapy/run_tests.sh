#!/bin/bash
# Convenience script to run lunapy tests

set -e  # Exit on error

echo "=================================================="
echo "lunapy Test Suite"
echo "=================================================="
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "Error: pytest not found"
    echo "Install with: pip install pytest pytest-cov"
    exit 1
fi

# Check if lunapy is installed
python -c "import lunapy" 2>/dev/null || {
    echo "Error: lunapy not installed"
    echo "Install with: pip install -e ."
    exit 1
}

# Parse arguments
COVERAGE=false
VERBOSE=false
TEST_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --coverage|-c)
            COVERAGE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Usage: ./run_tests.sh [OPTIONS] [TEST_FILE]"
            echo ""
            echo "Options:"
            echo "  --coverage, -c    Generate coverage report"
            echo "  --verbose, -v     Verbose output"
            echo "  --help, -h        Show this help"
            echo ""
            echo "Examples:"
            echo "  ./run_tests.sh                          # Run all tests"
            echo "  ./run_tests.sh --coverage               # Run with coverage"
            echo "  ./run_tests.sh tests/test_basic.py      # Run specific file"
            echo "  ./run_tests.sh -v -c                    # Verbose with coverage"
            exit 0
            ;;
        *)
            TEST_FILE="$1"
            shift
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="pytest"

if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -vv"
else
    PYTEST_CMD="$PYTEST_CMD -v"
fi

if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=lunapy --cov-report=term --cov-report=html"
fi

if [ -n "$TEST_FILE" ]; then
    PYTEST_CMD="$PYTEST_CMD $TEST_FILE"
else
    PYTEST_CMD="$PYTEST_CMD tests/"
fi

# Run tests
echo "Running: $PYTEST_CMD"
echo ""

$PYTEST_CMD

# Show coverage report location if generated
if [ "$COVERAGE" = true ]; then
    echo ""
    echo "=================================================="
    echo "Coverage report generated: htmlcov/index.html"
    echo "=================================================="
fi

echo ""
echo "Tests complete!"
