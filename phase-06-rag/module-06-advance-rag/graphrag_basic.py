# graphrag_basic.py


# Documents from our knowledge base
documents = {
    "D1": "Home loan applicants need identity proof and address proof.",
    "D2": "Home loan applicants should submit salary slips and bank statements.",
    "D3": "Home loan approval depends on the applicant's credit score.",
    "D4": "Personal loans can be used for various personal expenses."
}


# Entities and their relationships
graph = {
    "Home Loan": {
        "requires": [
            "Identity Proof",
            "Address Proof",
            "Salary Slips",
            "Bank Statements"
        ],
        "depends_on": [
            "Credit Score"
        ]
    }
}


# Find entities mentioned in the query
def find_entities(query):
    """Find known graph entities that appear in the user query."""

    query = query.lower()

    entities = []

    if "home loan" in query:
        entities.append("Home Loan")

    if "personal loan" in query:
        entities.append("Personal Loan")

    return entities


# Retrieve related information from the graph
def retrieve_from_graph(entity):
    """Retrieve relationships connected to the selected entity."""

    if entity not in graph:
        return {}

    return graph[entity]


# User query
query = "What does a home loan require?"


print("=" * 70)
print("USER QUERY")
print("=" * 70)

print(query)


# Step 1: Identify entities
entities = find_entities(query)


print("\n" + "=" * 70)
print("IDENTIFIED ENTITIES")
print("=" * 70)

for entity in entities:
    print(entity)


# Step 2: Traverse the graph
print("\n" + "=" * 70)
print("GRAPH RETRIEVAL")
print("=" * 70)

for entity in entities:

    relationships = retrieve_from_graph(entity)

    for relation, targets in relationships.items():

        for target in targets:

            print(
                f"{entity} --{relation}--> {target}"
            )


# Step 3: Build context for the LLM
print("\n" + "=" * 70)
print("RETRIEVED CONTEXT")
print("=" * 70)

for entity in entities:

    relationships = retrieve_from_graph(entity)

    for relation, targets in relationships.items():

        for target in targets:

            print(
                f"{entity} {relation} {target}"
            )