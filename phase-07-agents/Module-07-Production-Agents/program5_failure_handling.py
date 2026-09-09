import requests
import time

# Maximum number of retry attempts for temporary failures
MAX_RETRIES = 3

# Initial delay used for exponential backoff
INITIAL_DELAY = 1


def classify_failure(status_code):
    # Classify the API failure based on HTTP status code
    if status_code in [500, 502, 503, 504]:
        return "TEMPORARY"

    # 400-level errors are generally permanent client-side failures
    if status_code in [400, 401, 403, 404]:
        return "PERMANENT"

    # Anything else is treated as unknown
    return "UNKNOWN"


def call_api(url):
    # Try the API up to the configured retry limit
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\nAPI Attempt {attempt}/{MAX_RETRIES}")

        try:
            # Call the external API
            response = requests.get(url, timeout=5)

            # Return immediately when the API succeeds
            if response.status_code == 200:
                return "SUCCESS", response.json()

            # Classify the HTTP failure
            failure_type = classify_failure(response.status_code)

            print(f"HTTP Status: {response.status_code}")
            print(f"Failure Type: {failure_type}")

            # Retry only temporary failures
            if failure_type == "TEMPORARY" and attempt < MAX_RETRIES:
                delay = INITIAL_DELAY * (2 ** (attempt - 1))
                print(f"Temporary failure. Retrying in {delay} seconds...")
                time.sleep(delay)
                continue

            # Stop immediately for permanent failures
            if failure_type == "PERMANENT":
                return "PERMANENT_FAILURE", None

            # Escalate unknown failures
            return "ESCALATE", None

        except requests.RequestException as error:
            # Handle network-level failures
            print(f"Network Error: {error}")

            # Retry network failures while attempts remain
            if attempt < MAX_RETRIES:
                delay = INITIAL_DELAY * (2 ** (attempt - 1))
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                continue

            # Escalate after all network retries fail
            return "ESCALATE", None

    # Return escalation if all attempts are exhausted
    return "ESCALATE", None


# Use a real public API that returns HTTP 500
api_url = "https://httpbin.org/status/500"

# Call the API and classify the result
status, result = call_api(api_url)


# Handle the successful API path
if status == "SUCCESS":
    print("\nFinal Result:")
    print(result)

# Handle permanent failures
elif status == "PERMANENT_FAILURE":
    print("\nFinal Action:")
    print("Permanent failure. Do not retry.")

# Handle escalation
elif status == "ESCALATE":
    print("\nFinal Action:")
    print("Failure could not be recovered. Escalate to operations team.")