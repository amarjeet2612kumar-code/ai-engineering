from typing import Any  # Provides flexible type hints for tool arguments.
from dotenv import load_dotenv  # Loads environment variables from the .env file.
from openai import OpenAI  # Provides the OpenAI API client.


load_dotenv()  # Loads OPENAI_API_KEY from the .env file.

client = OpenAI()  # Creates the OpenAI API client.

MODEL_NAME = "gpt-5-mini"  # Defines the LLM used by the agent.


# Defines which tools the current user is allowed to use.
USER_TOOL_PERMISSIONS = {
    "check_spark_job": True,
    "read_spark_logs": True,
    "restart_spark_job": True,
    "kill_spark_job": False,
}


# Defines the resources the current user can access.
USER_RESOURCE_PERMISSIONS = {
    "dev": True,
    "staging": True,
    "production": False,
}


# Defines the risk level associated with each action.
ACTION_RISK = {
    "check_spark_job": "LOW",
    "read_spark_logs": "LOW",
    "restart_spark_job": "HIGH",
    "kill_spark_job": "CRITICAL",
}


# Defines tools that can modify or terminate workloads.
DESTRUCTIVE_TOOLS = {
    "restart_spark_job",
    "kill_spark_job",
}


def check_tool_authorization(tool_name):
    # Checks whether the user has permission to use the requested tool.
    allowed = USER_TOOL_PERMISSIONS.get(tool_name, False)

    # Returns a denial when the tool is not authorized.
    if not allowed:
        return {
            "allowed": False,
            "reason": f"User is not authorized to use tool: {tool_name}",
        }

    # Returns success when tool authorization passes.
    return {
        "allowed": True,
        "reason": "Tool authorization passed",
    }


def check_resource_authorization(resource):
    # Checks whether the user can access the requested resource environment.
    allowed = USER_RESOURCE_PERMISSIONS.get(resource, False)

    # Fails closed when the resource is unknown or unauthorized.
    if not allowed:
        return {
            "allowed": False,
            "reason": f"User is not authorized to access resource: {resource}",
        }

    # Returns success when resource authorization passes.
    return {
        "allowed": True,
        "reason": "Resource authorization passed",
    }


def check_risk_policy(tool_name, resource):
    # Gets the risk level associated with the requested tool.
    risk = ACTION_RISK.get(tool_name, "CRITICAL")

    # Blocks unknown actions by default.
    if tool_name not in ACTION_RISK:
        return {
            "decision": "BLOCK",
            "risk": "CRITICAL",
            "reason": "Unknown tool - fail closed",
        }

    # Blocks destructive actions against production.
    if resource == "production" and tool_name in DESTRUCTIVE_TOOLS:
        return {
            "decision": "BLOCK",
            "risk": risk,
            "reason": "Destructive production action is blocked",
        }

    # Requires human approval for high-risk actions.
    if risk == "HIGH":
        return {
            "decision": "APPROVAL_REQUIRED",
            "risk": risk,
            "reason": "High-risk action requires human approval",
        }

    # Blocks critical-risk actions by default.
    if risk == "CRITICAL":
        return {
            "decision": "BLOCK",
            "risk": risk,
            "reason": "Critical-risk action is blocked",
        }

    # Allows low-risk actions.
    return {
        "decision": "ALLOW",
        "risk": risk,
        "reason": "Low-risk action allowed",
    }


def execute_tool(tool_name, arguments):
    # Simulates tool execution after all security checks pass.
    print("\nTOOL EXECUTION")
    print("-" * 70)

    # Prints the tool selected by the application.
    print(f"Tool: {tool_name}")

    # Prints the validated arguments.
    print(f"Arguments: {arguments}")

    # Returns a simulated tool result.
    return {
        "executed": True,
        "tool": tool_name,
        "result": "Simulated tool execution successful",
    }


def secure_tool_pipeline(tool_name, arguments):
    # Prints the requested tool.
    print("\nREQUESTED TOOL")
    print("-" * 70)
    print(tool_name)

    # Prints the requested arguments.
    print("\nREQUESTED ARGUMENTS")
    print("-" * 70)
    print(arguments)

    # Extracts the target resource from the request.
    resource = arguments.get("resource", "unknown")

    # Runs tool-level authorization.
    tool_result = check_tool_authorization(tool_name)

    # Prints the tool authorization decision.
    print("\nTOOL AUTHORIZATION")
    print("-" * 70)
    print(tool_result)

    # Stops processing when tool authorization fails.
    if not tool_result["allowed"]:
        return {
            "status": "BLOCK",
            "stage": "TOOL_AUTHORIZATION",
            "reason": tool_result["reason"],
        }

    # Runs resource-level authorization.
    resource_result = check_resource_authorization(resource)

    # Prints the resource authorization decision.
    print("\nRESOURCE AUTHORIZATION")
    print("-" * 70)
    print(resource_result)

    # Stops processing when resource authorization fails.
    if not resource_result["allowed"]:
        return {
            "status": "BLOCK",
            "stage": "RESOURCE_AUTHORIZATION",
            "reason": resource_result["reason"],
        }

    # Runs the risk policy after authorization succeeds.
    risk_result = check_risk_policy(tool_name, resource)

    # Prints the risk decision.
    print("\nRISK POLICY")
    print("-" * 70)
    print(risk_result)

    # Stops execution when human approval is required.
    if risk_result["decision"] == "APPROVAL_REQUIRED":
        return {
            "status": "APPROVAL_REQUIRED",
            "stage": "RISK_POLICY",
            "reason": risk_result["reason"],
        }

    # Stops execution when the action is blocked.
    if risk_result["decision"] == "BLOCK":
        return {
            "status": "BLOCK",
            "stage": "RISK_POLICY",
            "reason": risk_result["reason"],
        }

    # Executes the tool only after every guardrail passes.
    tool_execution = execute_tool(tool_name, arguments)

    # Returns the successful execution result.
    return {
        "status": "ALLOW",
        "stage": "TOOL_EXECUTION",
        "reason": "All action guardrails passed",
        "tool_result": tool_execution,
    }


def run_test(test_name, tool_name, arguments):
    # Prints the test name.
    print("\n" + "=" * 70)
    print(test_name)
    print("=" * 70)

    # Runs the secure tool pipeline.
    result = secure_tool_pipeline(tool_name, arguments)

    # Prints the final security decision.
    print("\nFINAL RESULT")
    print("-" * 70)
    print(result)


def main():
    # Tests a low-risk operation against an authorized development resource.
    run_test(
        "TEST 1 - READ DEVELOPMENT JOB",
        "check_spark_job",
        {
            "job_id": "spark-dev-001",
            "resource": "dev",
        },
    )

    # Tests reading logs from an authorized staging resource.
    run_test(
        "TEST 2 - READ STAGING LOGS",
        "read_spark_logs",
        {
            "job_id": "spark-stage-001",
            "resource": "staging",
        },
    )

    # Tests a high-risk restart operation.
    run_test(
        "TEST 3 - RESTART STAGING JOB",
        "restart_spark_job",
        {
            "job_id": "spark-stage-002",
            "resource": "staging",
        },
    )

    # Tests an unauthorized tool.
    run_test(
        "TEST 4 - UNAUTHORIZED KILL TOOL",
        "kill_spark_job",
        {
            "job_id": "spark-prod-001",
            "resource": "production",
        },
    )

    # Tests resource-level authorization independently.
    run_test(
        "TEST 5 - AUTHORIZED TOOL BUT UNAUTHORIZED PRODUCTION RESOURCE",
        "read_spark_logs",
        {
            "job_id": "spark-prod-002",
            "resource": "production",
        },
    )

    # Tests an unknown tool to verify fail-closed behavior.
    run_test(
        "TEST 6 - UNKNOWN TOOL",
        "delete_cluster",
        {
            "cluster": "production-cluster",
            "resource": "production",
        },
    )


if __name__ == "__main__":
    # Starts the test suite when the program is executed directly.
    main()