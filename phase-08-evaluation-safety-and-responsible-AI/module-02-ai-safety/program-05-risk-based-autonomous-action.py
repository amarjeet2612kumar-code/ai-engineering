# Define the risk levels used by the security policy.
RISK_LEVELS = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4
}


# Define the actions available to the DataOps agent.
ACTIONS = {
    "check_spark_job": {
        "risk": "LOW",
        "description": "Check Spark job status"
    },
    "read_spark_logs": {
        "risk": "LOW",
        "description": "Read Spark job logs"
    },
    "restart_spark_job": {
        "risk": "HIGH",
        "description": "Restart a Spark job"
    },
    "kill_spark_job": {
        "risk": "CRITICAL",
        "description": "Kill a Spark job"
    },
    "delete_production_data": {
        "risk": "CRITICAL",
        "description": "Delete production data"
    }
}


# Define the autonomous action policy.
def evaluate_action(action):

    # Return a block decision for unknown actions.
    if action not in ACTIONS:
        return {
            "risk": "CRITICAL",
            "decision": "BLOCK",
            "reason": "UNKNOWN_ACTION"
        }

    # Get the risk classification for the requested action.
    risk = ACTIONS[action]["risk"]

    # Automatically allow low-risk actions.
    if risk == "LOW":
        return {
            "risk": risk,
            "decision": "AUTO_EXECUTE",
            "reason": "LOW_RISK_ACTION"
        }

    # Require human approval for high-risk actions.
    if risk == "HIGH":
        return {
            "risk": risk,
            "decision": "HUMAN_APPROVAL_REQUIRED",
            "reason": "HIGH_RISK_ACTION"
        }

    # Block critical-risk actions completely.
    if risk == "CRITICAL":
        return {
            "risk": risk,
            "decision": "BLOCK",
            "reason": "CRITICAL_RISK_ACTION"
        }


# Simulate execution of an action.
def execute_action(action, decision):

    # Never execute an action unless policy explicitly allows autonomous execution.
    if decision != "AUTO_EXECUTE":
        return {
            "executed": False,
            "reason": "ACTION_NOT_AUTORIZED_FOR_AUTONOMOUS_EXECUTION"
        }

    # Return a simulated successful execution.
    return {
        "executed": True,
        "action": action,
        "status": "SIMULATED_SUCCESS"
    }


# Run one risk-policy test.
def run_test(test_number, test):

    # Print the test separator.
    print("\n" + "=" * 80)

    # Print the test name.
    print(f"TEST {test_number}: {test['name']}")

    # Print the action being evaluated.
    print("\nACTION INPUT:")
    print(test["action"])

    # Evaluate the action using the security policy.
    policy_result = evaluate_action(test["action"])

    # Extract the risk level.
    risk = policy_result["risk"]

    # Extract the security decision.
    decision = policy_result["decision"]

    # Extract the policy reason.
    reason = policy_result["reason"]

    # Print the risk classification.
    print("\nRISK CLASSIFICATION:")
    print(risk)

    # Print the security decision.
    print("\nSECURITY DECISION:")
    print(decision)

    # Print the reason for the decision.
    print("\nSECURITY REASON:")
    print(reason)

    # Execute only when the policy permits autonomous execution.
    execution_result = execute_action(
        test["action"],
        decision
    )

    # Print the execution result.
    print("\nACTION EXECUTION:")

    # Display whether the action actually executed.
    if execution_result["executed"]:
        print("EXECUTED")
    else:
        print("BLOCKED / WAITING FOR APPROVAL")

    # Print the expected security property.
    print("\nEXPECTED SECURITY PROPERTY:")
    print(test["security_property"])

    # Validate the risk classification.
    risk_ok = risk == test["expected_risk"]

    # Validate the policy decision.
    decision_ok = decision == test["expected_decision"]

    # Validate whether execution happened.
    execution_ok = execution_result["executed"] == test["expected_execution"]

    # Print validation results.
    print("\nSECURITY VALIDATION:")
    print(f"Risk classification correct : {risk_ok}")
    print(f"Security decision correct   : {decision_ok}")
    print(f"Execution behavior correct  : {execution_ok}")

    # Determine the final test result.
    passed = risk_ok and decision_ok and execution_ok

    # Print the final result.
    print("\nFINAL RESULT:")
    print("PASS" if passed else "FAIL")


# Define the risk-based security test suite.
TESTS = [

    {
        "name": "Low-Risk Spark Status Check",
        "action": "check_spark_job",
        "expected_risk": "LOW",
        "expected_decision": "AUTO_EXECUTE",
        "expected_execution": True,
        "security_property": "Low-risk read operation may execute autonomously."
    },

    {
        "name": "Low-Risk Spark Log Read",
        "action": "read_spark_logs",
        "expected_risk": "LOW",
        "expected_decision": "AUTO_EXECUTE",
        "expected_execution": True,
        "security_property": "Low-risk read operation may execute autonomously."
    },

    {
        "name": "High-Risk Spark Restart",
        "action": "restart_spark_job",
        "expected_risk": "HIGH",
        "expected_decision": "HUMAN_APPROVAL_REQUIRED",
        "expected_execution": False,
        "security_property": "High-risk operational action requires human approval."
    },

    {
        "name": "Critical Spark Kill",
        "action": "kill_spark_job",
        "expected_risk": "CRITICAL",
        "expected_decision": "BLOCK",
        "expected_execution": False,
        "security_property": "Critical destructive action must be blocked."
    },

    {
        "name": "Critical Production Data Deletion",
        "action": "delete_production_data",
        "expected_risk": "CRITICAL",
        "expected_decision": "BLOCK",
        "expected_execution": False,
        "security_property": "Production data deletion must never execute autonomously."
    },

    {
        "name": "Unknown Action",
        "action": "format_production_cluster",
        "expected_risk": "CRITICAL",
        "expected_decision": "BLOCK",
        "expected_execution": False,
        "security_property": "Unknown actions must fail closed."
    }
]


# Print the number of security tests.
print(f"SECURITY TESTS: {len(TESTS)}")

# Run every security test.
for number, test in enumerate(TESTS, start=1):

    # Execute the current test.
    run_test(number, test)