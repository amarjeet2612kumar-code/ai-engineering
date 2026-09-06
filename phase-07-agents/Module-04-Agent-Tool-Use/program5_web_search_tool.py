import os  # Import os to read environment variables.
import ollama  # Import Ollama for communicating with the local LLM.
from dotenv import load_dotenv  # Import dotenv to load the API key from .env.
from tavily import TavilyClient  # Import Tavily's official Python client.


load_dotenv()  # Load variables from the .env file.

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))  # Create the Tavily client using the API key.


def search_web(query: str) -> str:  # Define the web search tool.
    response = tavily.search(
        query=query,  # Send the search query to Tavily.
        max_results=5,  # Limit the number of search results.
    )

    results = []  # Create a list to store formatted search results.

    for item in response["results"]:  # Process every result returned by Tavily.
        results.append(
            f"Title: {item['title']}\n"
            f"URL: {item['url']}\n"
            f"Content: {item['content']}\n"
        )

    return "\n".join(results)  # Return all search results as one string.


tools = [
    {
        "type": "function",  # Define this capability as a callable function.
        "function": {
            "name": "search_web",  # Give the tool its name.
            "description": "Search the web for current or external information.",  # Tell the LLM when to use the tool.
            "parameters": {
                "type": "object",  # Define the function arguments as an object.
                "properties": {
                    "query": {
                        "type": "string",  # Define the search query as a string.
                        "description": "The information to search for on the web.",  # Explain the query.
                    }
                },
                "required": ["query"],  # Require the query parameter.
            },
        },
    }
]


user_request = "What is the latest Python version?"  # Define a question requiring current information.


messages = [
    {
        "role": "user",  # Identify this as the user's message.
        "content": user_request,  # Store the user's question.
    }
]


response = ollama.chat(
    model="llama3.2:3b",  # Use the local LLM.
    messages=messages,  # Send the conversation.
    tools=tools,  # Give the LLM access to the web search tool.
)


assistant_message = response["message"]  # Extract the assistant message.

messages.append(assistant_message)  # Add the assistant message to the conversation.

tool_calls = assistant_message.get("tool_calls", [])  # Get the tool calls selected by the LLM.


if tool_calls:  # Check whether the LLM selected the web search tool.

    tool_call = tool_calls[0]  # Get the first tool call.

    tool_name = tool_call["function"]["name"]  # Extract the selected tool name.

    arguments = tool_call["function"]["arguments"]  # Extract the tool arguments.

    print("Selected Tool:", tool_name)  # Display the selected tool.

    print("Search Query:", arguments["query"])  # Display the search query.

    result = search_web(**arguments)  # Execute the Tavily search.

    print("\nSearch Results:\n")  # Display a heading for the search results.

    print(result)  # Display the retrieved search information.

    messages.append(
        {
            "role": "tool",  # Identify this message as a tool result.
            "content": result,  # Send Tavily results back to the LLM.
        }
    )

    response = ollama.chat(
        model="llama3.2:3b",  # Use the local LLM again.
        messages=messages,  # Send the original question and search results.
    )

    print("\nFinal Answer:", response["message"]["content"])  # Display the final answer.

else:  # Handle cases where the LLM does not select a tool.
    print("No web search tool was selected.")  # Display that no tool was used.
    print("LLM Response:", assistant_message.get("content", ""))  # Display the LLM response.