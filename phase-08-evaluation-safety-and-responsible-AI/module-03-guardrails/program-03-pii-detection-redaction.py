from dotenv import load_dotenv  # Loads environment variables from the .env file.
from openai import OpenAI  # Provides the OpenAI API client.
from presidio_analyzer import AnalyzerEngine  # Detects PII entities in text.
from presidio_anonymizer import AnonymizerEngine  # Redacts detected PII entities.


load_dotenv()  # Loads OPENAI_API_KEY from the .env file.

client = OpenAI()  # Creates the OpenAI API client.

MODEL_NAME = "gpt-5-mini"  # Defines the LLM used after PII sanitization.

analyzer = AnalyzerEngine()  # Creates the Presidio PII analyzer.

anonymizer = AnonymizerEngine()  # Creates the Presidio anonymization engine.


def detect_pii(text):
    # Runs Presidio against the input text.
    results = analyzer.analyze(
        text=text,
        language="en",
    )

    # Returns all detected PII entities.
    return results


def redact_pii(text, results):
    # Replaces detected PII values with anonymized placeholders.
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
    )

    # Returns the sanitized text.
    return anonymized.text


def call_llm(safe_text):
    # Sends only the sanitized text to the LLM.
    response = client.responses.create(
        model=MODEL_NAME,
        instructions=(
            "You are a secure DataOps Copilot. "
            "Answer the user's technical question clearly. "
            "Do not attempt to reconstruct or guess redacted personal information."
        ),
        input=safe_text,
    )

    # Returns the generated response.
    return response.output_text


def run_test(test_name, user_input):
    # Prints the test name.
    print("\n" + "=" * 70)
    print(test_name)
    print("=" * 70)

    # Prints the original user input.
    print("\nORIGINAL INPUT")
    print("-" * 70)
    print(user_input)

    # Detects PII using Presidio.
    pii_results = detect_pii(user_input)

    # Prints the detected PII entities.
    print("\nDETECTED PII")
    print("-" * 70)

    if not pii_results:
        print("No PII detected.")
    else:
        for result in pii_results:
            print({
                "entity": result.entity_type,
                "start": result.start,
                "end": result.end,
                "score": round(result.score, 3),
            })

    # Redacts the detected PII.
    safe_text = redact_pii(user_input, pii_results)

    # Prints the sanitized text.
    print("\nSANITIZED INPUT")
    print("-" * 70)
    print(safe_text)

    # Sends sanitized input to the LLM.
    llm_output = call_llm(safe_text)

    # Prints the LLM response.
    print("\nLLM OUTPUT")
    print("-" * 70)
    print(llm_output)


def main():
    # Tests an email address.
    run_test(
        "TEST 1 - EMAIL",
        "Send the Spark report to amarjeet@example.com.",
    )

    # Tests a phone number.
    run_test(
        "TEST 2 - PHONE",
        "The customer phone number is 9876543210.",
    )

    # Tests a person's name.
    run_test(
        "TEST 3 - PERSON",
        "Contact Rahul Sharma regarding the failed Spark job.",
    )

    # Tests an IP address.
    run_test(
        "TEST 4 - IP ADDRESS",
        "The Spark executor is running on IP 192.168.10.25.",
    )

    # Tests multiple PII entities in the same request.
    run_test(
        "TEST 5 - MULTIPLE PII",
        (
            "Customer Rahul Sharma can be contacted at "
            "rahul.sharma@example.com or 9876543210."
        ),
    )

    # Tests normal technical input without obvious PII.
    run_test(
        "TEST 6 - NO PII",
        "Explain why Spark executors can run out of memory.",
    )


if __name__ == "__main__":
    # Starts the test suite when the program is executed directly.
    main()