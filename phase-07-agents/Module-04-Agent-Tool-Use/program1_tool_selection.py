import ollama  # Import the Ollama Python library.

def get_payment_status(transaction_id):  # Define the payment status tool.
    return f"Payment {transaction_id} is FAILED."  # Return the payment status.

def get_customer_details(customer_id):  # Define the customer details tool.
    return f"Customer details for {customer_id} are available."  # Return customer details.

def get_account_balance(customer_id):  # Define the account balance tool.
    return f"Account balance for {customer_id} is ₹5,000."  # Return the account balance.

tools = {  # Create a dictionary containing the available tools.
    "get_payment_status": get_payment_status,  # Register the payment status tool.
    "get_customer_details": get_customer_details,  # Register the customer details tool.
    "get_account_balance": get_account_balance  # Register the account balance tool.
}  # Finish the tools dictionary.

user_request = "Check the status of payment transaction 1001."  # Define the user's request.

prompt = f"""  # Create the tool-selection prompt.
You are an AI agent that selects tools.  # Tell the model its role.

User request: {user_request}  # Provide the user's request.

Available tools:  # List the available tools.
- get_payment_status: checks payment transaction status.  # Describe the first tool.
- get_customer_details: gets customer information.  # Describe the second tool.
- get_account_balance: gets account balance.  # Describe the third tool.

Choose the best tool for the request.  # Ask the model to make the tool-selection decision.

Your response must contain this exact format:  # Define a simple output format.
TOOL: tool_name  # Tell the model how to identify the selected tool.
"""  # Finish the prompt.

response = ollama.chat(  # Send the prompt to Ollama.
    model="llama3.2:3b",  # Use the local llama3.2 model.
    messages=[{"role": "user", "content": prompt}]  # Send the prompt to the model.
)  # Finish the Ollama request.

llm_response = response["message"]["content"].strip()  # Extract the model's response.

print("LLM Response:")  # Print a heading for the model response.
print(llm_response)  # Display the complete model response.

selected_tool = None  # Start with no selected tool.

for tool_name in tools:  # Check each available tool name.
    if tool_name in llm_response:  # Check whether the tool name appears in the model response.
        selected_tool = tool_name  # Store the matching tool name.
        break  # Stop searching after finding the tool.

print("Selected Tool:", selected_tool)  # Display the selected tool.

if selected_tool == "get_payment_status":  # Check whether the payment status tool was selected.
    result = tools[selected_tool]("1001")  # Execute the payment status tool.
elif selected_tool == "get_customer_details":  # Check whether the customer details tool was selected.
    result = tools[selected_tool]("1001")  # Execute the customer details tool.
elif selected_tool == "get_account_balance":  # Check whether the account balance tool was selected.
    result = tools[selected_tool]("1001")  # Execute the account balance tool.
else:  # Handle the case where no valid tool was selected.
    result = "No valid tool was selected."  # Store an error message.

print("Tool Result:", result)  # Display the result returned by the tool.