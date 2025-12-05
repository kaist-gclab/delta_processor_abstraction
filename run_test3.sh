#!/usr/bin/env bash
set -euo pipefail

echo "==========================="
echo "Start Test 3: Mesh Abstraction Area of Original Mesh Over 70%"
echo "==========================="

fail=0  # track if any class < 70

# Read lines from python, unbuffered
while read -r line; do
    echo "$line"  # still show original output

    # Match lines like: "Intersection over Class 1: 98.666%"
    if [[ "$line" =~ ^Intersection\ over\ Class\ ([0-9]+):\ ([0-9.]+)% ]]; then
        cls="${BASH_REMATCH[1]}"
        val="${BASH_REMATCH[2]}"   # e.g. "98.666"

        # Compare float using bc
        if (( $(echo "$val < 70.0" | bc -l) )); then
            echo "FAIL: Class $cls Abstraction Area $val < 70%"
            fail=1
        else
            echo "PASS: Class $cls Abstraction Area $val >= 70%"
        fi
    fi
done < <(python -u mesh_abstraction.py)

echo "==========================="
echo "Result of Abstraction Area Test"

if [[ "$fail" -eq 0 ]]; then
    echo "PASS: All Class Abstraction Area >= 70%"
else
    echo "FAIL: At least one Class Abstraction Area < 70%"
fi

echo "End of Test 3"
echo "==========================="