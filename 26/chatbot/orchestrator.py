# orchestrator.py
from agents.websearch_agent import run_websearch_agent
from agents.matlab_executor_agent import run_matlab_executor_agent
from dotenv import load_dotenv
import os
from openai import OpenAI
import json
import base64

load_dotenv()

agent = OpenAI(
    base_url=f"{os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')}/v1",
    api_key="ollama"
)

tools = [
  {
    "type": "function",
    "function": {
      "name": "route_query",
      "description": "Routes the query to the appropriate handler based on the type",
      "parameters": {
        "type": "object",
        "properties": {
          "type": {
            "type": "string",
            "enum": ["web_search", "matlab_executor"],
            "description": "The target handler for the query"
          },
          "query": {
            "type": "string",
            "description": "The actual query to be processed"
          }
        },
        "required": ["type", "query"]
      }
    }
  }
]

def classify_query(user_query, image_base64=None, conversation_history=None):
    """Classify user query using simple keyword-based method. Supports images and conversation history."""
    # Build user message content - add image if present
    user_content = f"{user_query}"
    if image_base64:
        user_content = [
            {
                "type": "text",
                "text": user_query
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                }
            }
        ]
    
    # Build messages list with system prompt, conversation history, and current query
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert agent that routes user queries to the appropriate handler. "
                "If an image is provided, analyze it and use it to help classify the query. "
                "Use the conversation history to understand context and follow-up questions. "
                "\n\n"
                "Route queries as follows:\n"
                "1. type = 'matlab_executor': For general MATLAB programming tasks, like any kind of calcuations to be performed, "
                "plotting, simulations, power flow analysis, or any task requiring custom MATLAB code execution. This includes transfer functions, "
                "step responses, bode plots, state-space models, differential equations, Ybus, bus voltages, admittance matrices, etc.\n"
                "2. type = 'web_search': For general knowledge questions not related to technical computation.\n"
                "\n"
                "For small talk and greetings, DO NOT call any tool - just respond naturally."
            )
        }
    ]
    
    # Add conversation history if available
    if conversation_history:
        messages.extend(conversation_history)
    
    # Add current user query
    messages.append({
        "role": "user",
        "content": user_content
    })
    
    respone = agent.chat.completions.create(
        model=os.environ.get("OLLAMA_MODEL", "llama3"),
        messages=messages,
        tools=tools,
        max_tokens=8000,
        stream=False
    )
    
    if respone.choices[0].message.tool_calls:
        tool_response = respone.choices[0].message.tool_calls[0].function
        print(tool_response)
        if tool_response.name == "route_query":
            arguments = json.loads(tool_response.arguments)
            print(arguments)
            return arguments['type'], arguments['query']
        else:
            return {"error": "Unexpected tool call"}
    else:
        response_content = respone.choices[0].message.content
        if response_content:
            return response_content.strip(), None
        return None, None

def contextualize_matlab_query(user_query, conversation_history=None):
    """
    Rewrites the user's query into a standalone, data-rich MATLAB task 
    by filling in context and tables from the conversation history.
    """
    if not conversation_history:
        return user_query
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert MATLAB task contextualizer. "
                "Your goal is to rewrite the user's current query into a standalone, comprehensive "
                "instruction for a MATLAB executor agent. "
                "\n\n"
                "Rules:\n"
                "1. If the user refers to data (like 'the system', 'those values', 'the table') "
                "that was provided in previous messages, you MUST find that data and include "
                "it explicitly in the rewritten query (e.g., as Markdown tables or lists).\n"
                "2. If the user's query is a follow-up (e.g., 'now plot the result'), "
                "rewrite it to include the original task context so the agent knows what to plot.\n"
                "3. Ensure all technical parameters (bus data, line data, impedances, etc.) mentioned "
                "previously are preserved in the rewritten prompt.\n"
                "4. If there is an image in the history, take its description into account if relevant.\n"
                "5. Output ONLY the rewritten prompt. No explanation or conversation."
            )
        }
    ]
    
    # Add history
    messages.extend(conversation_history)
    
    # Add current query
    messages.append({
        "role": "user",
        "content": user_query
    })
    
    response = agent.chat.completions.create(
        model=os.environ.get("OLLAMA_MODEL", "llama3"),
        messages=messages,
        max_tokens=8000,
        stream=False
    )
    
    rewritten_query = response.choices[0].message.content
    return rewritten_query.strip() if rewritten_query else user_query

def orchestrate(user_query, image_base64=None, csv_files: list = None, conversation_history=None):
    """Orchestrate query handling with optional image, multiple CSV files, and conversation history support.
    
    csv_files: list of dicts with keys 'path' and 'preview', one per CSV file.
               e.g. [{"path": "/data/a.csv", "preview": "col1,col2\\n1,2\\n..."}, ...]
    """
    answer, query = classify_query(user_query, image_base64, conversation_history)
    print(f"Classified query as: {answer}")
    if answer == "web_search":
        return run_websearch_agent(query)
    elif answer == "matlab_executor":
        # Fill the query with context/data from history before execution
        contextualized_prompt = contextualize_matlab_query(user_query, conversation_history)
        print(f"Contextualized MATLAB Prompt: {contextualized_prompt}")
        return run_matlab_executor_agent(contextualized_prompt, csv_files)
    else:
        return answer

def main():
    while True:
        user_query = input("User: ")
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        if not user_query:
            print("Please enter a valid query.")
            continue
        try:
            result = orchestrate(user_query)
            print(f"Response: {result}\n")
        except Exception as e:
            import sys
            error_msg = str(e).encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
            print(f"Error: {error_msg}")

if __name__ == "__main__":
    main()