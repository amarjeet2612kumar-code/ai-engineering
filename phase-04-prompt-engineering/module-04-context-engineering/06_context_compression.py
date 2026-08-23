import ollama


# =========================================================
# LARGE INVESTIGATION CONTEXT
#
# This represents information accumulated during
# a DataOps investigation.
# =========================================================

investigation_context = """
DataOps Incident Investigation

Job ID: CRM360-123

The Spark job started at 10:15 AM.

The job processes customer transaction data.

The job contains several stages including:
- reading transaction data
- joining customer information
- aggregating transaction records
- writing the final output

The Spark application initially started successfully.

The driver submitted the application successfully.

Executor 1 started successfully.

Executor 2 started successfully.

Executor 3 started successfully.

Executor 4 started successfully.

Executor 5 started successfully.

The application continued processing records.

Executor 3 reported high memory usage.

Executor 3 reported memory utilization of 91%.

A few minutes later Executor 3 reported memory utilization of 94%.

Executor 3 then reported memory utilization of 96%.

Executor 3 exceeded its configured memory limit.

The executor was removed from the application.

Spark reported:

ExecutorLostFailure

The executor exceeded the configured memory limit.

The Spark application entered a failed state.

The configured executor memory is 16 GB.

The executor memory overhead is configured to 2 GB.

The job processes a large amount of transaction data.

The aggregation stage contains a large shuffle.

The shuffle stage generated significant intermediate data.

Several log lines repeated the same ExecutorLostFailure message.

Several log lines repeated that Executor 3 exceeded memory.

The job failed after approximately 27 minutes.

The previous run completed successfully.

The previous successful run processed a smaller dataset.

The current dataset is approximately 2.5 times larger.

No Kafka errors were observed.

No Airflow scheduler errors were observed.

The Airflow task correctly detected that the Spark application failed.

The database connection was successful.

The output destination was available.

The failure occurred during the Spark aggregation stage.

The primary error observed was ExecutorLostFailure.

Memory utilization was significantly higher than in the previous run.

The investigation currently indicates a possible executor memory pressure problem.

Further investigation should determine whether the root cause is:
- insufficient executor memory
- excessive shuffle
- data skew
- an inefficient aggregation
- another memory-related problem

The final diagnosis should be based on available evidence rather than assumptions.
"""


print("========== ORIGINAL CONTEXT ==========")

print(investigation_context)


print(
    "\nOriginal character count:",
    len(investigation_context)
)


# =========================================================
# COMPRESS THE CONTEXT
#
# We ask the LLM to preserve information that is useful
# for continuing the investigation.
# =========================================================

compression_prompt = f"""
You are compressing context for a DataOps investigation.

Original investigation context:

{investigation_context}

Create a compact investigation summary.

Preserve:

1. Job ID
2. Failure
3. Important error
4. Important metrics
5. Relevant configuration
6. Evidence about what happened
7. Important comparison with the previous run
8. Systems that were ruled out
9. Remaining hypotheses
10. Important uncertainty

Remove:

- repeated log messages
- irrelevant details
- redundant statements
- information that does not affect diagnosis

Do not invent information.

The compressed context must preserve the evidence needed
for another LLM call to continue the investigation.
"""


response = ollama.chat(
    model="llama3.2:3b",

    messages=[
        {
            "role": "user",
            "content": compression_prompt
        }
    ]
)


compressed_context = response["message"]["content"]


# =========================================================
# DISPLAY RESULTS
# =========================================================

print("\n\n========== COMPRESSED CONTEXT ==========")

print(compressed_context)


print(
    "\nCompressed character count:",
    len(compressed_context)
)


# =========================================================
# CALCULATE REDUCTION
# =========================================================

original_size = len(investigation_context)

compressed_size = len(compressed_context)

reduction = (
    1 - (compressed_size / original_size)
) * 100


print(
    "\nApproximate compression:",
    f"{reduction:.2f}%"
)