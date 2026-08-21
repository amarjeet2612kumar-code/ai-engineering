import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

text = "Explain Apache Spark partitioning."

tokens = encoding.encode(text)

print("Token count:", len(tokens))
print("Token IDs:", tokens)

for token in tokens:
    print(token, "→", encoding.decode([token]))