import os  # Provides access to environment variables.
import json  # Converts JSON text into Python dictionaries.
import time  # Provides a small delay between retries.

from groq import Groq  # Provides the Groq API client.
from dotenv import load_dotenv  # Loads variables from the .env file.


# ======================================================================
# ENVIRONMENT
# ======================================================================

load_dotenv()  # Load variables from the .env file.

GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Read the Groq API key.

if not GROQ_API_KEY:  # Check whether the API key exists.
    raise ValueError("GROQ_API_KEY is missing from .env")  # Stop execution if missing.


# Create the Groq client.
client = Groq(
    api_key=GROQ_API_KEY
)


# ======================================================================
# MODELS
# ======================================================================

GENERATOR_MODEL = "qwen/qwen3.6-27b"  # Model used to generate candidate answers.

JUDGE_MODEL = "openai/gpt-oss-20b"  # Independent model used to evaluate answers.


# ======================================================================
# QUALITY THRESHOLDS
# ======================================================================

MIN_AVG_CORRECTNESS = 4.0  # Minimum acceptable average correctness.

MIN_AVG_RELEVANCE = 4.0  # Minimum acceptable average relevance.

MIN_AVG_CLARITY = 4.0  # Minimum acceptable average clarity.

MIN_AVG_OVERALL = 4.0  # Minimum acceptable average overall score.

MIN_CRITICAL_CORRECTNESS = 3  # Minimum correctness allowed for critical cases.


# ======================================================================
# EVALUATION DATASET
# ======================================================================

evaluation_dataset = [
    {
        "id": 1,
        "question": "What is Apache Spark?",
        "critical": True,
    },
    {
        "id": 2,
        "question": "What is Apache Kafka?",
        "critical": True,
    },
    {
        "id": 3,
        "question": "What is Apache Airflow?",
        "critical": True,
    },
    {
        "id": 4,
        "question": "What is a Kafka topic?",
        "critical": False,
    },
    {
        "id": 5,
        "question": "What is a Spark DataFrame?",
        "critical": False,
    },
    {
        "id": 6,
        "question": "What is an Airflow DAG?",
        "critical": False,
    },
]


# ======================================================================
# GENERATOR RESPONSE CLEANING
# ======================================================================

def clean_generator_response(answer):
    # Remove leading and trailing whitespace.
    answer = answer.strip()

    # Handle a complete reasoning block.
    if "<think>" in answer and "</think>" in answer:

        # Keep only the content after the reasoning block.
        answer = answer.split("</think>", 1)[1].strip()

    # Handle an incomplete reasoning block.
    elif "<think>" in answer:

        # Keep only content before the reasoning block.
        answer = answer.split("<think>", 1)[0].strip()

    # Remove accidental markdown code fences.
    answer = answer.replace("```text", "").replace("```", "").strip()

    # Return the cleaned answer.
    return answer


# ======================================================================
# GENERATOR
# ======================================================================

def generate_answer(question):
    # Build a short prompt to reduce token usage.
    prompt = f"""
Answer this technical question accurately:

{question}

Rules:
- Return only the final answer.
- Do not show reasoning.
- Keep the answer below 100 words.
- Do not invent facts.
"""

    # Retry generation up to three times.
    for attempt in range(1, 4):

        try:
            # Call the Qwen generator with reasoning disabled.
            response = client.chat.completions.create(
                model=GENERATOR_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a concise technical assistant. "
                            "Return only the final answer."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0,
                max_tokens=200,
                reasoning_effort="none",
                reasoning_format="hidden",
            )

            # Extract the model response.
            raw_answer = response.choices[0].message.content or ""

            # Clean reasoning tags and formatting.
            answer = clean_generator_response(raw_answer)

            # Return the answer if it is not empty.
            if answer:
                return answer

            # Display diagnostic information.
            print(
                f"Generator attempt {attempt}: empty final answer."
            )

        except Exception as error:

            # Display the actual API error.
            print(
                f"Generator attempt {attempt} failed: {error}"
            )

        # Wait before retrying.
        time.sleep(2)

    # Stop execution if all attempts fail.
    raise RuntimeError(
        f"Generator model '{GENERATOR_MODEL}' "
        "did not return a usable answer after 3 attempts."
    )


# ======================================================================
# JUDGE
# ======================================================================

def judge_answer(question, answer):
    # Build a compact evaluation prompt.
    prompt = f"""
Evaluate the following technical answer.

QUESTION:
{question}

ANSWER:
{answer}

Evaluate these criteria from 1 to 5.

CORRECTNESS:
1 = incorrect
3 = partially correct
5 = fully correct

RELEVANCE:
1 = irrelevant
3 = partially relevant
5 = directly relevant

CLARITY:
1 = unclear
3 = acceptable
5 = very clear

OVERALL:
Give an overall score from 1 to 5.

Evaluation rules:
- Check technical facts carefully.
- Penalize factual errors.
- Do not reward unnecessary length.
- Do not penalize concise answers.
- Do not use answer length as a quality signal.
- Return only valid JSON.
- Do not return markdown.
- Do not return reasoning.

Return exactly this structure:

{{
    "correctness": 1,
    "relevance": 1,
    "clarity": 1,
    "overall": 1.0,
    "reason": "short explanation"
}}
"""

    # Retry judging up to three times.
    for attempt in range(1, 4):

        try:
            # Send the candidate answer to the independent judge.
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict technical evaluator. "
                            "Return only valid JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0,
                max_tokens=300,
                reasoning_effort="low",
                include_reasoning=False,
                response_format={
                    "type": "json_object"
                },
            )

            # Extract the judge response.
            raw_judge = response.choices[0].message.content or ""

            # Remove leading and trailing whitespace.
            raw_judge = raw_judge.strip()

            # Reject an empty judge response.
            if not raw_judge:
                raise ValueError(
                    "Judge returned an empty response."
                )

            # Parse the JSON response.
            score = json.loads(raw_judge)

            # Define the fields that must exist.
            required_fields = [
                "correctness",
                "relevance",
                "clarity",
                "overall",
                "reason",
            ]

            # Validate required fields.
            for field in required_fields:

                # Raise an error when a field is missing.
                if field not in score:
                    raise ValueError(
                        f"Missing field: {field}"
                    )

            # Validate correctness score.
            if not 1 <= float(score["correctness"]) <= 5:
                raise ValueError(
                    "Invalid correctness score."
                )

            # Validate relevance score.
            if not 1 <= float(score["relevance"]) <= 5:
                raise ValueError(
                    "Invalid relevance score."
                )

            # Validate clarity score.
            if not 1 <= float(score["clarity"]) <= 5:
                raise ValueError(
                    "Invalid clarity score."
                )

            # Validate overall score.
            if not 1 <= float(score["overall"]) <= 5:
                raise ValueError(
                    "Invalid overall score."
                )

            # Return the validated evaluation.
            return score

        except Exception as error:

            # Display the judge failure.
            print(
                f"Judge attempt {attempt} failed: {error}"
            )

        # Wait before retrying.
        time.sleep(2)

    # Stop execution after all judge attempts fail.
    raise RuntimeError(
        f"Judge model '{JUDGE_MODEL}' "
        "failed after 3 attempts."
    )


# ======================================================================
# RUN EVALUATION
# ======================================================================

results = []  # Store results for all test cases.


print("=" * 70)  # Print the program header.

print("EVALUATION QUALITY GATE")  # Display the program title.

print("=" * 70)  # Print the separator.

print(
    f"Generator Model : {GENERATOR_MODEL}"
)  # Display the generator model.

print(
    f"Judge Model     : {JUDGE_MODEL}"
)  # Display the judge model.


# Process every evaluation case.
for test_case in evaluation_dataset:

    # Read the test case ID.
    test_id = test_case["id"]

    # Read the evaluation question.
    question = test_case["question"]

    # Read whether the case is critical.
    critical = test_case["critical"]

    print("\n" + "=" * 70)  # Print the test separator.

    print(f"TEST CASE {test_id}")  # Display the test case number.

    print("=" * 70)  # Print the test separator.

    print("\nQuestion:")  # Display the question heading.

    print(question)  # Display the question.

    print("\n--- GENERATED ANSWER ---")  # Display answer heading.

    # Generate the candidate answer.
    answer = generate_answer(question)

    # Display the generated answer.
    print(answer)

    print("\n--- JUDGE SCORE ---")  # Display judge heading.

    # Evaluate the generated answer.
    score = judge_answer(
        question,
        answer
    )

    # Display the structured evaluation.
    print(
        json.dumps(
            score,
            indent=4
        )
    )

    # Save the result.
    results.append(
        {
            "id": test_id,
            "question": question,
            "critical": critical,
            "answer": answer,
            "score": score,
        }
    )


# ======================================================================
# AGGREGATE METRICS
# ======================================================================

# Collect correctness scores.
correctness_scores = [
    float(result["score"]["correctness"])
    for result in results
]

# Collect relevance scores.
relevance_scores = [
    float(result["score"]["relevance"])
    for result in results
]

# Collect clarity scores.
clarity_scores = [
    float(result["score"]["clarity"])
    for result in results
]

# Collect overall scores.
overall_scores = [
    float(result["score"]["overall"])
    for result in results
]


# Calculate average correctness.
avg_correctness = (
    sum(correctness_scores)
    / len(correctness_scores)
)

# Calculate average relevance.
avg_relevance = (
    sum(relevance_scores)
    / len(relevance_scores)
)

# Calculate average clarity.
avg_clarity = (
    sum(clarity_scores)
    / len(clarity_scores)
)

# Calculate average overall score.
avg_overall = (
    sum(overall_scores)
    / len(overall_scores)
)


# ======================================================================
# CRITICAL CASE CHECK
# ======================================================================

critical_failures = []  # Store critical cases that fail correctness.


# Check every evaluation result.
for result in results:

    # Only evaluate critical test cases.
    if result["critical"]:

        # Read the correctness score.
        correctness = float(
            result["score"]["correctness"]
        )

        # Check the critical-case threshold.
        if correctness < MIN_CRITICAL_CORRECTNESS:

            # Store the failed case ID.
            critical_failures.append(
                result["id"]
            )


# ======================================================================
# QUALITY GATE
# ======================================================================

# Check the average correctness threshold.
correctness_pass = (
    avg_correctness >= MIN_AVG_CORRECTNESS
)

# Check the average relevance threshold.
relevance_pass = (
    avg_relevance >= MIN_AVG_RELEVANCE
)

# Check the average clarity threshold.
clarity_pass = (
    avg_clarity >= MIN_AVG_CLARITY
)

# Check the average overall threshold.
overall_pass = (
    avg_overall >= MIN_AVG_OVERALL
)

# Check critical-case correctness.
critical_pass = (
    len(critical_failures) == 0
)


# Require every quality condition to pass.
quality_gate_passed = (
    correctness_pass
    and relevance_pass
    and clarity_pass
    and overall_pass
    and critical_pass
)


# ======================================================================
# FINAL REPORT
# ======================================================================

print("\n" + "=" * 70)  # Print the final report separator.

print("QUALITY GATE REPORT")  # Display the report title.

print("=" * 70)  # Print the final report separator.


print(
    f"\nAverage Correctness : {avg_correctness:.2f}"
)  # Display average correctness.

print(
    f"Average Relevance   : {avg_relevance:.2f}"
)  # Display average relevance.

print(
    f"Average Clarity     : {avg_clarity:.2f}"
)  # Display average clarity.

print(
    f"Average Overall     : {avg_overall:.2f}"
)  # Display average overall score.


print("\nTHRESHOLD CHECKS")  # Display threshold section.


print(
    f"Correctness >= {MIN_AVG_CORRECTNESS}: "
    f"{'PASS' if correctness_pass else 'FAIL'}"
)  # Display correctness threshold result.


print(
    f"Relevance >= {MIN_AVG_RELEVANCE}: "
    f"{'PASS' if relevance_pass else 'FAIL'}"
)  # Display relevance threshold result.


print(
    f"Clarity >= {MIN_AVG_CLARITY}: "
    f"{'PASS' if clarity_pass else 'FAIL'}"
)  # Display clarity threshold result.


print(
    f"Overall >= {MIN_AVG_OVERALL}: "
    f"{'PASS' if overall_pass else 'FAIL'}"
)  # Display overall threshold result.


print(
    f"Critical correctness >= {MIN_CRITICAL_CORRECTNESS}: "
    f"{'PASS' if critical_pass else 'FAIL'}"
)  # Display critical-case threshold result.


# Display critical failures when present.
if critical_failures:

    print(
        f"\nCritical failures: {critical_failures}"
    )

else:

    print("\nCritical failures: None")


print("\n" + "=" * 70)  # Print final separator.


# Display the final quality decision.
if quality_gate_passed:

    print("✅ QUALITY GATE PASSED")

else:

    print("❌ QUALITY GATE FAILED")


print("=" * 70)  # Print final separator.