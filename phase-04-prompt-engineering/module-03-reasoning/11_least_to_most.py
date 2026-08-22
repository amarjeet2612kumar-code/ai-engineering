import ollama


# =========================================================
# Main problem
# =========================================================

main_problem = """
We need to migrate a production data pipeline from
an on-prem Hadoop environment to AWS.

The pipeline uses:

- HDFS
- Hive
- Spark
- Kafka

Design a practical migration plan covering:

1. Data migration
2. Spark migration
3. Hive migration
4. Kafka migration
5. Validation
6. Cutover
"""


# =========================================================
# STEP 1
# Decompose the complex problem
# =========================================================

decomposition_prompt = f"""
We need to solve this complex problem:

{main_problem}

Break this problem into a small number of
independent or logically ordered subproblems.

Return only the numbered subproblems.
"""


response = ollama.chat(
    model="llama3.2:3b",

    messages=[
        {
            "role": "user",
            "content": decomposition_prompt
        }
    ]
)


subproblems_text = response["message"]["content"]


print("========== SUBPROBLEMS ==========")

print(subproblems_text)


# =========================================================
# STEP 2
# Manually define the subproblems for this learning demo
#
# We do this so the execution flow is deterministic.
# =========================================================

subproblems = [
    "What AWS services should replace the current HDFS and Hive storage layer?",

    "How should the existing Spark workloads be migrated to AWS?",

    "How should Kafka workloads be migrated or integrated with AWS?",

    "How should data and pipeline correctness be validated?",

    "How should production cutover and rollback be handled?"
]


# =========================================================
# STEP 3
# Solve each subproblem sequentially
# =========================================================

solutions = []


for index, subproblem in enumerate(
    subproblems,
    start=1
):

    prompt = f"""
We are solving a larger migration problem.

Main problem:

{main_problem}


Previously solved subproblems:

{solutions}


Current subproblem:

{subproblem}

Solve only this subproblem.

Provide a practical and concise answer.
"""


    response = ollama.chat(
        model="llama3.2:3b",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


    solution = response["message"]["content"].strip()


    solutions.append(
        {
            "subproblem": subproblem,
            "solution": solution
        }
    )


    print(
        f"\n========== SUBPROBLEM {index} =========="
    )

    print(subproblem)

    print("\nSolution:")

    print(solution)


# =========================================================
# STEP 4
# Combine all subproblem solutions
# =========================================================

final_prompt = f"""
Create a final migration plan for the following problem:

{main_problem}

The following subproblems have already been solved:

{solutions}

Combine these solutions into one coherent,
ordered migration plan.

Include:

1. Migration phases
2. Dependencies
3. Validation
4. Cutover
5. Rollback considerations

Do not introduce unrelated recommendations.
"""


response = ollama.chat(
    model="llama3.2:3b",

    messages=[
        {
            "role": "user",
            "content": final_prompt
        }
    ]
)


print("\n\n========== FINAL MIGRATION PLAN ==========")

print(
    response["message"]["content"]
)