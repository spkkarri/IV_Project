"""
Available models: ['codestral:22b', 'phi4:latest', 'llava:latest', 'qwen2.5:14b-instruct', 'deepseek-r1:14b', 
'deepseek-coder:6.7b-instruct', 'mannix/deepseek-coder-v2-lite-instruct:latest', 'mxbai-embed-large:latest', 
'nomic-embed-text:latest', 'phi4-reasoning:latest', 'qwen2.5vl:32b', 'devstral:24b', 'deepseek-r1:latest', 
'deepseek-coder-v2:latest', 'llama3.2:latest']
"""

import requests
import json
import time

REMOTE_HOST = "https://drivers-hygiene-gotta-modern.trycloudflare.com"
MODEL_NAME = "qwen2.5:14b-instruct"

# Define a tool schema for the model to use
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a given symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "The stock symbol, e.g. AAPL"}
                },
                "required": ["symbol"]
            }
        }
    }
]

def test_performance_and_tools():
    url = f"{REMOTE_HOST}/api/chat"
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "What is the stock price of NVDA?"}],
        "tools": TOOLS,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        # 1. Test Tool Calling
        message = data.get("message", {})
        if "tool_calls" in message:
            print("🛠️ Tool Call Detected!")
            for call in message["tool_calls"]:
                print(f"   Function: {call['function']['name']}")
                print(f"   Arguments: {call['function']['arguments']}")
        else:
            print("📝 Normal Response (No tool triggered):")
            print(message.get("content"))

        # 2. Test Throughput (TPS)
        # Ollama provides duration in nanoseconds (10^9)
        eval_count = data.get("eval_count", 0)
        eval_duration_ns = data.get("eval_duration", 1)
        
        if eval_count > 0:
            tps = (eval_count / eval_duration_ns) * 1_000_000_000
            print(f"\n🚀 Throughput Stats:")
            print(f"   Output Tokens: {eval_count}")
            print(f"   Generation Rate: {tps:.2f} tokens/s")
            print(f"   Total Time: {data.get('total_duration', 0) / 1e9:.2f}s")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_performance_and_tools()