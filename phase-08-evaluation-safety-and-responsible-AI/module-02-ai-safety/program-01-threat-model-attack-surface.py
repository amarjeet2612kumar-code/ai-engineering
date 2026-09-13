# Store the important assets that our AI application needs to protect.
assets = [
    "Production Spark Jobs",
    "Kafka Data",
    "Customer Data",
    "RAG Documents",
    "API Credentials",
]

# Store the people or systems that could perform an attack.
threat_actors = [
    "Malicious User",
    "Compromised User Account",
    "Malicious Document",
]

# Store the places where an attacker can interact with our AI application.
attack_surface = [
    "User Prompt",
    "RAG Documents",
    "Agent Tools",
    "Application API",
]

# Store possible threats that we identified during threat modeling.
threats = [
    {
        "name": "Prompt Injection",
        "entry_point": "User Prompt",
        "asset": "Production Spark Jobs",
        "impact": "Unauthorized production action",
        "likelihood": 4,
        "impact_score": 5,
        "mitigation": "Input validation and tool authorization",
    },
    {
        "name": "RAG Poisoning",
        "entry_point": "RAG Documents",
        "asset": "RAG Documents",
        "impact": "AI receives malicious information",
        "likelihood": 3,
        "impact_score": 4,
        "mitigation": "Trusted sources and document validation",
    },
    {
        "name": "Data Leakage",
        "entry_point": "RAG Documents",
        "asset": "Customer Data",
        "impact": "Sensitive data exposed",
        "likelihood": 3,
        "impact_score": 5,
        "mitigation": "Authorization and access-controlled retrieval",
    },
    {
        "name": "Tool Abuse",
        "entry_point": "Agent Tools",
        "asset": "Production Spark Jobs",
        "impact": "Unauthorized job operation",
        "likelihood": 3,
        "impact_score": 5,
        "mitigation": "Tool allowlist and human approval",
    },
]

# Print the title of the threat model report.
print("\n=== DATAOPS COPILOT THREAT MODEL ===")

# Print all assets that need protection.
print("\n1. ASSETS")

# Loop through every asset.
for asset in assets:
    # Display the current asset.
    print(f"- {asset}")

# Print all possible threat actors.
print("\n2. THREAT ACTORS")

# Loop through every threat actor.
for actor in threat_actors:
    # Display the current threat actor.
    print(f"- {actor}")

# Print all possible attack entry points.
print("\n3. ATTACK SURFACE")

# Loop through every attack surface entry.
for entry_point in attack_surface:
    # Display the current entry point.
    print(f"- {entry_point}")

# Print the detailed threat analysis.
print("\n4. THREAT ANALYSIS")

# Loop through every identified threat.
for threat in threats:
    # Calculate risk using likelihood multiplied by impact.
    risk = threat["likelihood"] * threat["impact_score"]

    # Print the name of the current threat.
    print(f"\nThreat       : {threat['name']}")

    # Print where the attacker can enter the system.
    print(f"Entry Point  : {threat['entry_point']}")

    # Print which asset could be affected.
    print(f"Asset        : {threat['asset']}")

    # Print the possible business or technical impact.
    print(f"Impact       : {threat['impact']}")

    # Print the likelihood score.
    print(f"Likelihood   : {threat['likelihood']}/5")

    # Print the impact score.
    print(f"Impact Score : {threat['impact_score']}/5")

    # Print the calculated risk score.
    print(f"Risk Score   : {risk}/25")

    # Print the recommended security control.
    print(f"Mitigation   : {threat['mitigation']}")

# Print the attack paths identified in our architecture.
print("\n5. ATTACK PATHS")

# Describe the first attack path from user input to production.
print("1. User → Prompt Injection → LLM → Agent → Spark Tool → Production")

# Describe the second attack path through the RAG system.
print("2. Malicious Document → RAG → LLM → Agent → Tool")

# Describe the third attack path for sensitive information.
print("3. User → RAG → Unauthorized Document → LLM → Data Leakage")

# Print the final security principle.
print("\n6. SECURITY PRINCIPLE")

# Explain that the LLM should not be the final authorization authority.
print("LLM requests an action; the application decides whether the action is allowed.")