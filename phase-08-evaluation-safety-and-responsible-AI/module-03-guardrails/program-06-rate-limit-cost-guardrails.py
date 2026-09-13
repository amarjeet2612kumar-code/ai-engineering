# Load environment variables from the .env file.
from dotenv import load_dotenv

# Load variables from .env into the environment.
load_dotenv()


# Maximum number of requests a user can make.
MAX_REQUESTS = 3

# Maximum input tokens allowed for a single request.
MAX_INPUT_TOKENS = 1000

# Maximum estimated cost allowed for a single request.
MAX_REQUEST_COST = 0.05

# Maximum cumulative cost allowed for a user.
MAX_USER_COST = 0.10


# Store request counts for users.
USER_REQUEST_COUNTS = {}

# Store cumulative estimated cost for users.
USER_TOTAL_COST = {}


# Check whether the user has exceeded the request rate limit.
def check_rate_limit(user_id):
    # Get the current number of requests for the user.
    request_count = USER_REQUEST_COUNTS.get(user_id, 0)

    # Block the request when the user has reached the limit.
    if request_count >= MAX_REQUESTS:
        return {
            "allowed": False,
            "reason": "Rate limit exceeded",
        }

    # Allow the request when the limit has not been reached.
    return {
        "allowed": True,
        "reason": "Rate limit check passed",
    }


# Check whether the request contains too many input tokens.
def check_token_limit(input_tokens):
    # Block requests exceeding the maximum input token limit.
    if input_tokens > MAX_INPUT_TOKENS:
        return {
            "allowed": False,
            "reason": f"Input token limit exceeded: {input_tokens}",
        }

    # Allow requests within the token limit.
    return {
        "allowed": True,
        "reason": "Token limit check passed",
    }


# Check whether the current request is too expensive.
def check_request_cost(request_cost):
    # Block requests exceeding the per-request cost limit.
    if request_cost > MAX_REQUEST_COST:
        return {
            "allowed": False,
            "reason": f"Request cost limit exceeded: ${request_cost:.4f}",
        }

    # Allow requests within the per-request cost limit.
    return {
        "allowed": True,
        "reason": "Request cost check passed",
    }


# Check whether the user has exceeded their cumulative cost budget.
def check_user_cost(user_id, request_cost):
    # Get the user's current accumulated cost.
    current_cost = USER_TOTAL_COST.get(user_id, 0.0)

    # Calculate the cost after allowing the current request.
    projected_cost = current_cost + request_cost

    # Block the request when the projected cost exceeds the budget.
    if projected_cost > MAX_USER_COST:
        return {
            "allowed": False,
            "reason": f"User cost budget exceeded: ${projected_cost:.4f}",
        }

    # Allow the request when the projected cost is within the budget.
    return {
        "allowed": True,
        "reason": "User cost budget check passed",
    }


# Record usage only after all guardrails allow the request.
def record_usage(user_id, request_cost):
    # Increase the user's request count.
    USER_REQUEST_COUNTS[user_id] = USER_REQUEST_COUNTS.get(user_id, 0) + 1

    # Increase the user's cumulative cost.
    USER_TOTAL_COST[user_id] = USER_TOTAL_COST.get(user_id, 0.0) + request_cost


# Run the complete rate and cost guardrail pipeline.
def secure_llm_pipeline(user_id, input_tokens, request_cost):
    # Display the requested user.
    print(f"USER: {user_id}")

    # Display the estimated input token count.
    print(f"INPUT TOKENS: {input_tokens}")

    # Display the estimated request cost.
    print(f"REQUEST COST: ${request_cost:.4f}")

    # Check the user's request rate.
    rate_result = check_rate_limit(user_id)

    # Display the rate-limit result.
    print("\nRATE LIMIT")
    print(rate_result)

    # Stop immediately when the rate limit fails.
    if not rate_result["allowed"]:
        return {
            "status": "BLOCK",
            "stage": "RATE_LIMIT",
            "reason": rate_result["reason"],
        }

    # Check the input token limit.
    token_result = check_token_limit(input_tokens)

    # Display the token-limit result.
    print("\nTOKEN LIMIT")
    print(token_result)

    # Stop immediately when the token limit fails.
    if not token_result["allowed"]:
        return {
            "status": "BLOCK",
            "stage": "TOKEN_LIMIT",
            "reason": token_result["reason"],
        }

    # Check the cost of the individual request.
    request_cost_result = check_request_cost(request_cost)

    # Display the request-cost result.
    print("\nREQUEST COST LIMIT")
    print(request_cost_result)

    # Stop immediately when the request cost is too high.
    if not request_cost_result["allowed"]:
        return {
            "status": "BLOCK",
            "stage": "REQUEST_COST",
            "reason": request_cost_result["reason"],
        }

    # Check the user's cumulative cost budget.
    user_cost_result = check_user_cost(user_id, request_cost)

    # Display the cumulative cost result.
    print("\nUSER COST BUDGET")
    print(user_cost_result)

    # Stop immediately when the user's budget would be exceeded.
    if not user_cost_result["allowed"]:
        return {
            "status": "BLOCK",
            "stage": "USER_COST_BUDGET",
            "reason": user_cost_result["reason"],
        }

    # Record usage only after every guardrail has passed.
    record_usage(user_id, request_cost)

    # Simulate successful LLM execution.
    print("\nLLM EXECUTION")
    print("Request sent to LLM")

    # Return the successful result.
    return {
        "status": "ALLOW",
        "stage": "LLM_EXECUTION",
        "reason": "All rate and cost guardrails passed",
    }


# Run one test case and compare it with the expected result.
def run_test(test_name, user_id, input_tokens, request_cost, expected_status):
    # Print the test heading.
    print("\n" + "=" * 70)
    print(test_name)
    print("=" * 70)

    # Run the secure LLM pipeline.
    result = secure_llm_pipeline(
        user_id=user_id,
        input_tokens=input_tokens,
        request_cost=request_cost,
    )

    # Display the final result.
    print("\nFINAL RESULT")
    print(result)

    # Compare the actual status with the expected status.
    passed = result["status"] == expected_status

    # Display the test result.
    print(f"\nTEST RESULT: {'PASS' if passed else 'FAIL'}")

    # Return whether the test passed.
    return passed


# Execute all practical test cases.
def main():
    # Track the number of successful tests.
    passed_tests = 0

    # Test 1: normal request should be allowed.
    if run_test(
        "TEST 1 - NORMAL REQUEST",
        "user-001",
        500,
        0.02,
        "ALLOW",
    ):
        passed_tests += 1

    # Test 2: another normal request should be allowed.
    if run_test(
        "TEST 2 - SECOND REQUEST",
        "user-001",
        600,
        0.02,
        "ALLOW",
    ):
        passed_tests += 1

    # Test 3: a third request should still be allowed.
    if run_test(
        "TEST 3 - THIRD REQUEST",
        "user-001",
        700,
        0.02,
        "ALLOW",
    ):
        passed_tests += 1

    # Test 4: fourth request should be blocked by rate limiting.
    if run_test(
        "TEST 4 - RATE LIMIT EXCEEDED",
        "user-001",
        500,
        0.01,
        "BLOCK",
    ):
        passed_tests += 1

    # Test 5: a new user with too many input tokens should be blocked.
    if run_test(
        "TEST 5 - TOKEN LIMIT EXCEEDED",
        "user-002",
        1500,
        0.02,
        "BLOCK",
    ):
        passed_tests += 1

    # Test 6: a request exceeding the per-request cost limit should be blocked.
    if run_test(
        "TEST 6 - REQUEST COST LIMIT EXCEEDED",
        "user-003",
        500,
        0.08,
        "BLOCK",
    ):
        passed_tests += 1

    # Test 7: a request that would exceed the user's cumulative budget should be blocked.
    USER_TOTAL_COST["user-004"] = 0.09

    # Run the cumulative budget test.
    if run_test(
        "TEST 7 - USER COST BUDGET EXCEEDED",
        "user-004",
        500,
        0.02,
        "BLOCK",
    ):
        passed_tests += 1

    # Display the final test summary.
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Passed: {passed_tests}/7")
    print(f"Gate: {'PASS' if passed_tests == 7 else 'FAIL'}")


# Start the program.
if __name__ == "__main__":
    main()