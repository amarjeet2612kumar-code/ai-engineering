import asyncio
import json
import re
from pathlib import Path
from typing import TypedDict

import ollama
from langgraph.graph import StateGraph, START, END
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# Get the directory containing this program.
BASE_DIR = Path(__file__).resolve().parent

# Build the absolute path to the MCP server.
SERVER_FILE = BASE_DIR / "mcp_server.py"


# Define the state shared between LangGraph nodes.
class AgentState(TypedDict, total=False):
    question: str
    available_tools: list
    selected_tool: str
    job_id: int
    tool_result: dict | str
    answer: str


# Define how the MCP server should be started.
SERVER_PARAMS = StdioServerParameters(
    command="python",
    args=[str(SERVER_FILE)],
)


# Discover tools exposed by the MCP server.
async def discover_mcp_tools():

    # Start the MCP server as a subprocess.
    async with stdio_client(SERVER_PARAMS) as (read, write):

        # Create an MCP client session.
        async with ClientSession(read, write) as session:

            # Initialize the MCP connection.
            await session.initialize()

            # Ask the MCP server for its available tools.
            response = await session.list_tools()

            # Convert MCP tool objects into simple dictionaries.
            tools = []

            for tool in response.tools:
                tools.append(
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": tool.inputSchema,
                    }
                )

            # Return the discovered tools.
            return tools


# Execute a selected MCP tool.
async def execute_mcp_tool(tool_name: str, job_id: int):

    # Start the MCP server as a subprocess.
    async with stdio_client(SERVER_PARAMS) as (read, write):

        # Create an MCP client session.
        async with ClientSession(read, write) as session:

            # Initialize the MCP connection.
            await session.initialize()

            # Discover tools exposed by this server.
            response = await session.list_tools()

            # Build a set containing valid tool names.
            allowed_tools = {
                tool.name
                for tool in response.tools
            }

            # Prevent execution of an unknown tool.
            if tool_name not in allowed_tools:
                return {
                    "error": f"Tool '{tool_name}' is not available."
                }

            # Call the selected MCP tool.
            result = await session.call_tool(
                tool_name,
                {
                    "request": {
                        "job_id": job_id
                    }
                },
            )

            # Return structured MCP output when available.
            if result.structuredContent:
                return result.structuredContent

            # Return text output when structured output is unavailable.
            if result.content:
                return result.content[0].text

            # Return a safe fallback when the tool returns nothing.
            return {
                "error": "MCP tool returned no result."
            }


# LangGraph node that discovers MCP tools.
def discover_tools_node(state: AgentState):

    # Discover tools from the MCP server.
    tools = asyncio.run(
        discover_mcp_tools()
    )

    # Store the discovered tools in workflow state.
    return {
        "available_tools": tools
    }


# LangGraph node where Ollama selects the MCP tool.
def select_tool_node(state: AgentState):

    # Convert tool definitions into readable JSON.
    tools_text = json.dumps(
        state["available_tools"],
        indent=2,
    )

    # Build a strict tool-selection prompt.
    prompt = f"""
You are a DataOps assistant.

User question:
{state["question"]}

Available MCP tools:
{tools_text}

Select the ONE tool that best answers the user's question.

Rules:
- Return ONLY the exact tool name.
- Do not explain your answer.
- Do not return JSON.
- Do not return any other text.
"""

    # Ask the local Ollama model to select a tool.
    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    # Normalize the model output.
    model_output = response["message"]["content"].strip()

    # Build a lookup of valid MCP tool names.
    valid_tools = {
        tool["name"]
        for tool in state["available_tools"]
    }

    # Match the model output against the real tool names.
    selected_tool = None

    for tool_name in valid_tools:

        # Check whether the exact tool name appears in the model response.
        if tool_name.lower() in model_output.lower():
            selected_tool = tool_name
            break

    # Fail safely if no valid tool was selected.
    if selected_tool is None:
        selected_tool = "INVALID_TOOL"

    # Store the validated tool selection.
    return {
        "selected_tool": selected_tool
    }


# LangGraph node that extracts the Spark job ID.
def extract_job_id_node(state: AgentState):

    # Read the user's question.
    question = state["question"]

    # Find a number following the word "job".
    match = re.search(
        r"\bjob\s+(\d+)\b",
        question,
        re.IGNORECASE,
    )

    # Stop safely if no job ID exists.
    if not match:
        raise ValueError(
            "No Spark job ID found in the question."
        )

    # Convert the extracted job ID from string to integer.
    job_id = int(match.group(1))

    # Store the job ID in workflow state.
    return {
        "job_id": job_id
    }


# LangGraph node that executes the selected MCP tool.
def execute_tool_node(state: AgentState):

    # Prevent execution when the LLM selected an invalid tool.
    if state["selected_tool"] == "INVALID_TOOL":
        return {
            "tool_result": {
                "error": "LLM could not select a valid MCP tool."
            }
        }

    # Execute the selected MCP tool.
    result = asyncio.run(
        execute_mcp_tool(
            state["selected_tool"],
            state["job_id"],
        )
    )

    # Store the MCP result in workflow state.
    return {
        "tool_result": result
    }


# LangGraph node that generates the final answer.
def final_answer_node(state: AgentState):

    # Convert the MCP result into readable JSON.
    tool_result = json.dumps(
        state["tool_result"],
        indent=2,
        default=str,
    )

    # Build the final-answer prompt.
    prompt = f"""
You are a DataOps assistant.

User question:
{state["question"]}

MCP tool used:
{state["selected_tool"]}

MCP result:
{tool_result}

Answer the user's question using only the MCP result.

Rules:
- Do not invent information.
- Keep the answer concise.
"""

    # Ask Ollama to generate the final answer.
    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    # Store the generated answer.
    return {
        "answer": response["message"]["content"].strip()
    }


# Create the LangGraph state graph.
builder = StateGraph(AgentState)

# Register each workflow node.
builder.add_node(
    "discover_tools",
    discover_tools_node,
)

builder.add_node(
    "select_tool",
    select_tool_node,
)

builder.add_node(
    "extract_job_id",
    extract_job_id_node,
)

builder.add_node(
    "execute_tool",
    execute_tool_node,
)

builder.add_node(
    "final_answer",
    final_answer_node,
)


# Define the workflow execution order.
builder.add_edge(
    START,
    "discover_tools",
)

builder.add_edge(
    "discover_tools",
    "select_tool",
)

builder.add_edge(
    "select_tool",
    "extract_job_id",
)

builder.add_edge(
    "extract_job_id",
    "execute_tool",
)

builder.add_edge(
    "execute_tool",
    "final_answer",
)

builder.add_edge(
    "final_answer",
    END,
)


# Compile the LangGraph workflow.
graph = builder.compile()


# Run the workflow when this file is executed directly.
if __name__ == "__main__":

    # Define the user's request.
    initial_state: AgentState = {
        "question": "Show me the logs of Spark job 101."
    }

    # Execute the complete workflow.
    result = graph.invoke(
        initial_state
    )

    # Display the MCP tools discovered from the server.
    print("\n=== MCP TOOLS ===")

    for tool in result["available_tools"]:
        print(
            f"- {tool['name']}: "
            f"{tool['description']}"
        )

    # Display the tool selected by Ollama.
    print("\n=== SELECTED TOOL ===")
    print(
        result["selected_tool"]
    )

    # Display the extracted job ID.
    print("\n=== JOB ID ===")
    print(
        result["job_id"]
    )

    # Display the actual MCP tool result.
    print("\n=== MCP RESULT ===")
    print(
        result["tool_result"]
    )

    # Display the final LLM-generated answer.
    print("\n=== FINAL ANSWER ===")
    print(
        result["answer"]
    )