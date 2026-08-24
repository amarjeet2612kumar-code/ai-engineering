import torch
from sentence_transformers import SentenceTransformer


# =========================================================
# 1. Load embedding model
# =========================================================

model = SentenceTransformer(
    "jinaai/jina-embeddings-v2-base-en"
)


# =========================================================
# 2. Document
# =========================================================

document = """
Apache Spark is a distributed data processing engine.
Spark applications contain a driver and one or more executors.
The driver coordinates the application and schedules tasks.
Executors perform the actual computation on worker nodes.

When Spark processes large datasets, executors require sufficient
memory to hold intermediate data and execute tasks.
If an executor does not have enough memory, the application can
experience an out-of-memory error.

Increasing executor memory can help some workloads, but memory
problems can also be caused by large partitions or inefficient
transformations.
"""


# =========================================================
# 3. Define chunks
# =========================================================

chunks = [
    "Spark applications contain a driver and one or more executors.",
    "Executors perform the actual computation on worker nodes.",
    "If an executor does not have enough memory, the application can experience an out-of-memory error.",
    "Memory problems can also be caused by large partitions or inefficient transformations."
]


# =========================================================
# 4. Normal chunking approach
# =========================================================

normal_embeddings = model.encode(
    chunks,
    convert_to_tensor=True
)


print("\n========== NORMAL CHUNKING ==========\n")

print(
    "Number of chunks:",
    len(normal_embeddings)
)

print(
    "Embedding dimension:",
    normal_embeddings.shape[1]
)


# =========================================================
# 5. Late chunking
# =========================================================

# Tokenize the COMPLETE document
tokenized = model.tokenizer(
    document,
    return_tensors="pt",
    truncation=True
)


# Generate contextual token embeddings
with torch.no_grad():

    output = model[0].auto_model(
        **tokenized
    )

token_embeddings = output.last_hidden_state


print("\n========== TOKEN EMBEDDINGS ==========\n")

print(
    "Token embedding shape:",
    token_embeddings.shape
)


# =========================================================
# 6. Create token ranges for chunks
# =========================================================

encoded_chunks = []

for chunk in chunks:

    chunk_tokens = model.tokenizer(
        chunk,
        add_special_tokens=False
    )["input_ids"]

    encoded_chunks.append(chunk_tokens)


# Find each chunk inside the document token sequence
document_tokens = tokenized["input_ids"][0].tolist()


chunk_embeddings = []


for chunk_tokens in encoded_chunks:

    chunk_length = len(chunk_tokens)

    start_position = None


    for i in range(
        len(document_tokens) - chunk_length + 1
    ):

        if document_tokens[
            i:i + chunk_length
        ] == chunk_tokens:

            start_position = i
            break


    if start_position is None:

        print(
            "Could not locate chunk:",
            chunk[:50]
        )

        continue


    end_position = (
        start_position + chunk_length
    )


    # Extract contextual token embeddings
    chunk_token_embeddings = token_embeddings[
        0,
        start_position:end_position,
        :
    ]


    # Mean pooling
    chunk_embedding = (
        chunk_token_embeddings.mean(dim=0)
    )


    chunk_embeddings.append(
        chunk_embedding
    )


# =========================================================
# 7. Display results
# =========================================================

print("\n========== LATE CHUNKING ==========\n")

print(
    "Number of chunk embeddings:",
    len(chunk_embeddings)
)

print(
    "Embedding dimension:",
    chunk_embeddings[0].shape[0]
)