# Define the possible risk levels for an operation.
RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


# Define the policy for a guardrail that successfully executes.
def handle_guardrail_success(risk_level):
    # Allow low-risk operations when the guardrail passes.
    if risk_level == "LOW":
        return {
            "decision": "ALLOW",
            "reason": "Guardrail passed for low-risk operation",
        }

    # Allow medium-risk operations when the guardrail passes.
    if risk_level == "MEDIUM":
        return {
            "decision": "ALLOW",
            "reason": "Guardrail passed for medium-risk operation",
        }

    # Require approval for high-risk operations.
    if risk_level == "HIGH":
        return {
            "decision": "APPROVAL_REQUIRED",
            "reason": "High-risk operation requires human approval",
        }

    # Block critical operations even when other checks pass.
    if risk_level == "CRITICAL":
        return {
            "decision": "BLOCK",
            "reason": "Critical operation is blocked by policy",
        }

    # Fail closed for unknown risk levels.
    return {
        "decision": "BLOCK",
        "reason": "Unknown risk level - fail closed",
    }


# Define what happens when the guardrail itself fails.
def handle_guardrail_failure(risk_level):
    # Low-risk operations can use a controlled fail-open policy.
    if risk_level == "LOW":
        return {
            "decision": "ALLOW",
            "reason": "Guardrail failed, but low-risk operation uses fail-open policy",
        }

    # Medium-risk operations require human approval when the guardrail fails.
    if risk_level == "MEDIUM":
        return {
            "decision": "APPROVAL_REQUIRED",
            "reason": "Guardrail failed, medium-risk operation requires approval",
        }

    # High-risk operations are blocked when the guardrail fails.
    if risk_level == "HIGH":
        return {
            "decision": "BLOCK",
            "reason": "Guardrail failed, high-risk operation uses fail-closed policy",
        }

    # Critical operations are always blocked when the guardrail fails.
    if risk_level == "CRITICAL":
        return {
            "decision": "BLOCK",
            "reason": "Guardrail failed, critical operation is blocked",
        }

    # Unknown risk levels are blocked by default.
    return {
        "decision": "BLOCK",
        "reason": "Unknown risk level - fail closed",
    }


# Simulate a guardrail execution.
def run_guardrail(risk_level, guardrail_failed):
    # Simulate a successful guardrail execution.
    if not guardrail_failed:
        return {
            "status": "SUCCESS",
            "result": handle_guardrail_success(risk_level),
        }

    # Simulate a guardrail failure.
    return {
        "status": "FAILED",
        "result": handle_guardrail_failure(risk_level),
    }


# Run a single test case.
def run_test(test_name, risk_level, guardrail_failed, expected_decision):
    # Print the test heading.
    print("\n" + "=" * 70)
    print(test_name)
    print("=" * 70)

    # Display the risk level.
    print(f"RISK LEVEL: {risk_level}")

    # Display whether the guardrail failed.
    print(f"GUARDRAIL FAILED: {guardrail_failed}")

    # Execute the simulated guardrail.
    result = run_guardrail(
        risk_level=risk_level,
        guardrail_failed=guardrail_failed,
    )

    # Display the guardrail status.
    print("\nGUARDRAIL STATUS")
    print(result["status"])

    # Display the policy decision.
    print("\nPOLICY DECISION")
    print(result["result"])

    # Compare the actual decision with the expected decision.
    passed = result["result"]["decision"] == expected_decision

    # Display the test result.
    print(f"\nTEST RESULT: {'PASS' if passed else 'FAIL'}")

    # Return whether the test passed.
    return passed


# Execute all practical tests.
def main():
    # Track successful tests.
    passed_tests = 0

    # Test 1: low-risk successful guardrail.
    if run_test(
        "TEST 1 - LOW RISK GUARDRAIL SUCCESS",
        "LOW",
        False,
        "ALLOW",
    ):
        passed_tests += 1

    # Test 2: low-risk guardrail failure uses controlled fail-open behavior.
    if run_test(
        "TEST 2 - LOW RISK GUARDRAIL FAILURE",
        "LOW",
        True,
        "ALLOW",
    ):
        passed_tests += 1

    # Test 3: medium-risk guardrail success is allowed.
    if run_test(
        "TEST 3 - MEDIUM RISK GUARDRAIL SUCCESS",
        "MEDIUM",
        False,
        "ALLOW",
    ):
        passed_tests += 1

    # Test 4: medium-risk guardrail failure requires human approval.
    if run_test(
        "TEST 4 - MEDIUM RISK GUARDRAIL FAILURE",
        "MEDIUM",
        True,
        "APPROVAL_REQUIRED",
    ):
        passed_tests += 1

    # Test 5: high-risk guardrail success requires human approval.
    if run_test(
        "TEST 5 - HIGH RISK GUARDRAIL SUCCESS",
        "HIGH",
        False,
        "APPROVAL_REQUIRED",
    ):
        passed_tests += 1

    # Test 6: high-risk guardrail failure is blocked.
    if run_test(
        "TEST 6 - HIGH RISK GUARDRAIL FAILURE",
        "HIGH",
        True,
        "BLOCK",
    ):
        passed_tests += 1

    # Test 7: critical guardrail success is blocked.
    if run_test(
        "TEST 7 - CRITICAL RISK GUARDRAIL SUCCESS",
        "CRITICAL",
        False,
        "BLOCK",
    ):
        passed_tests += 1

    # Test 8: critical guardrail failure is blocked.
    if run_test(
        "TEST 8 - CRITICAL RISK GUARDRAIL FAILURE",
        "CRITICAL",
        True,
        "BLOCK",
    ):
        passed_tests += 1

    # Test 9: unknown risk level fails closed.
    if run_test(
        "TEST 9 - UNKNOWN RISK LEVEL",
        "UNKNOWN",
        True,
        "BLOCK",
    ):
        passed_tests += 1

    # Display the final summary.
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Passed: {passed_tests}/9")
    print(f"Gate: {'PASS' if passed_tests == 9 else 'FAIL'}")


# Start the program.
if __name__ == "__main__":
    main()