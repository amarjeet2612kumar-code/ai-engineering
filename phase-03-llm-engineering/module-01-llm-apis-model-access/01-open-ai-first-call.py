from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

response = client.responses.create(
    model="gpt-5-mini",
    input="Explain Apache Spark partitioning in one sentence."
)

print(response.output_text)