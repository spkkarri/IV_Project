import sys
import os

# Add the parent directory to sys.path so we can import agents
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chatbot.agents.websearch_agent import run_websearch_agent
import json

def test_iterative_search():
    print("Testing Iterative Search Agent...")
    query = "What is the relationship between Renewable Energy and Battery Storage in the Knowledge Graph and generally on the web?"
    
    # We expect this to run at least 1-2 iterations.
    # Note: This will attempt to call port 8000 for KG. 
    # If the API is not running, it will return errors for KG part but continue with Web.
    
    result = run_websearch_agent(query)
    
    print("\n--- Final Synthesized Result ---\n")
    print(result)
    print("\n-------------------------------\n")

if __name__ == "__main__":
    test_iterative_search()
