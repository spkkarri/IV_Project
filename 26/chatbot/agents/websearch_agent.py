from ddgs import DDGS
from openai import OpenAI
import json
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
import time

load_dotenv()

client = OpenAI(
    base_url=f"{os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')}/v1",
    api_key="ollama"
)
MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

def search_web(query, num_results=3):
    """
    Searches DuckDuckGo and then scrapes the full text from the URLs.
    No API Key required.
    """
    print(f"Searching DDG for: {query}...")
    
    links = []
    
    # --- 1. Get URLs from DuckDuckGo ---
    try:
        with DDGS() as ddgs:
            # ddgs.text() returns an iterator of results
            results = ddgs.text(query, max_results=num_results)
            if results:
                for result in results:
                    links.append(result['href'])
            else:
                print("No results found.")
                return []
    except Exception as e:
        print(f"DDG Search failed: {e}")
        return []

    full_data = []

    # --- 2. Visit each URL and extract text ---
    for url in links:
        try:
            print(f"Scraping: {url}")
            
            # Use a standard browser User-Agent to avoid being blocked
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            # Disable SSL warnings and bypass SSL validation for sites with bad certs
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            # Timeout is important so the script doesn't hang on slow sites
            response = requests.get(url, headers=headers, timeout=10, verify=False)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract text from paragraphs
                paragraphs = [p.get_text().strip() for p in soup.find_all('p')]
                
                # Join paragraphs and clean up whitespace
                full_text = '\n\n'.join([p for p in paragraphs if p])
                
                if full_text:
                    full_data.append({
                        'url': url,
                        'title': soup.title.string if soup.title else "No Title",
                        'content': full_text[:10000] + "..." # Limit text to prevent overload
                    })
            else:
                print(f"Skipped {url} (Status: {response.status_code})")
                
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")

    return full_data

def query_knowledge_graph(query):
    """
    Queries the Knowledge Graph API for information.
    """
    kg_url = os.environ['NEO4J_URL']
    try:
        print(f"Querying Knowledge Graph for: {query}...")
        response = requests.post(kg_url, params={"query": query}, timeout=200)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Status {response.status_code}")
            return {"error": f"Status {response.status_code}"}
    except Exception as e:
        print(e)
        return {"error": str(e)}

def format_kg_relationships(kg_results, seen_relationships):
    """
    Groups relationships by PDF URL and formats them for the context.
    Deduplicates using (source, relation, target, pdf_url, page_number).
    """
    relationships = kg_results.get('relationships', [])
    if not relationships:
        return ""
    
    url_to_rels = {}
    for r in relationships:
        # Create a unique key for deduplication
        rel_key = (
            r.get('source'), 
            r.get('relation'), 
            r.get('target'), 
            r.get('pdf_url'), 
            r.get('page_number')
        )
        
        if rel_key in seen_relationships:
            continue
        
        seen_relationships.add(rel_key)
        
        url = r.get('pdf_url', 'Unknown Source')
        if url not in url_to_rels:
            url_to_rels[url] = []
        url_to_rels[url].append(r)
    
    if not url_to_rels:
        return ""
        
    output = "Knowledge Graph Extraction (Relationships):\n"
    for url, rels in url_to_rels.items():
        output += f"PDF URL: {url}\n"
        for r in rels:
            output += f"- {r['source']} --({r['relation']})--> {r['target']}\n"
            output += f"  Page: {r.get('page_number')}\n"
    return output

def check_search_completeness(user_query, accumulated_results):
    """
    Uses LLM to determine if gathered info is sufficient or if a new query is needed.
    """
    context = f"User Question: {user_query}\n\nGathered Information:\n"
    seen_relationships = set()
    for i, res in enumerate(accumulated_results):
        context += f"--- Iteration {i+1} (Query: {res['query']}) ---\n"
        # Summarize web results
        web_titles = [w.get('title', 'No Title') for w in res.get('web', [])]
        context += f"Web Results: {', '.join(web_titles[:3])}\n"
        # Summarize KG results
        kg_text = format_kg_relationships(res.get('kg', {}), seen_relationships)
        if kg_text:
            context += kg_text + "\n"
            print(f"Added {len(res.get('kg', {}).get('relationships', []))} KG relationships to context (Iteration {i+1}).")

    prompt = (
        "Based on the gathered information, is the user's question fully answered? "
        "If not, what specific missing information should we search for next? "
        "Return a JSON object with: 'is_complete' (boolean), 'next_query' (string, empty if complete), 'reasoning' (string)."
    )

    messages = [
        {"role": "system", "content": "You are a search coordinator. Your goal is to be thorough. If details are missing, suggest a targeted sub-query. Respond only in JSON."},
        {"role": "user", "content": context + prompt}
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=1000
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Completeness check error: {e}")
        return {"is_complete": True, "next_query": "", "reasoning": "Error in check, stopping."}

def synthesize_answer_with_llm(user_query, all_iterations):
    """
    Synthesizes the final answer using all accumulated data.
    """
    context = f"Synthesize a comprehensive answer for: {user_query}\n\n"
    seen_relationships = set()
    for i, res in enumerate(all_iterations):
        context += f"--- Iteration {i+1} (Query: {res['query']}) ---\n"
        context += "Web Findings:\n"
        for w in res.get('web', []):
            if 'content' in w:
                context += f"- [{w['title']}]({w['url']}): {w['content'][:1500]}\n"
        
        kg_text = format_kg_relationships(res.get('kg', {}), seen_relationships)
        if kg_text:
            context += kg_text + "\n"
        context += "\n"

    context += (
        "\nProvide a deep, structured response. Cite sources if possible. "
        "At the end, state: 'This result was generated by an iterative agent using Web and KG search results.'"
    )

    messages = [
        {"role": "system", "content": "You are a professional research assistant."},
        {"role": "user", "content": context}
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=8000
    )
    return response.choices[0].message.content

def route_query_to_source(user_query):
    """
    Uses LLM tool call to determine if a query is related to NIT Andhra Pradesh.
    Returns True if it is about NIT Andhra Pradesh (use KG), False otherwise (use Web Search).
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "route_query",
                "description": "Determines if the user's query is related to NIT Andhra Pradesh.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "is_nit_ap_related": {
                            "type": "boolean",
                            "description": "True if the query is about NIT Andhra Pradesh, False otherwise."
                        }
                    },
                    "required": ["is_nit_ap_related"]
                }
            }
        }
    ]

    messages = [
        {"role": "system", "content": "You are a routing assistant. Decide if the user's query is about NIT Andhra Pradesh. Use the provided tool to output your decision."},
        {"role": "user", "content": user_query}
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "route_query"}},
            max_tokens=100
        )
        
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            arguments = json.loads(tool_call.function.arguments)
            return arguments.get('is_nit_ap_related', False)
        return False
    except Exception as e:
        print(f"Routing error: {e}")
        return False

def run_websearch_agent(user_query):
    """
    Iterative agent that searches both Web and KG up to 4 times.
    """
    print(f"Starting iterative search for: {user_query}")
    
    # Router at the top
    # Use LLM tool call to determine if query is about NIT Andhra Pradesh
    is_nit_query = route_query_to_source(user_query)
    
    all_iterations = []
    current_query = user_query
    
    for i in range(4):
        print(f"\n>>>> Iteration {i+1}/4 | Query: {current_query}")
        
        if is_nit_query:
            print("Router: Routing to Knowledge Graph ONLY.")
            web_results = []
            kg_results = query_knowledge_graph(current_query)
        else:
            print("Router: Routing to Web Search ONLY.")
            web_results = search_web(current_query)
            kg_results = {}
        
        iteration_data = {
            "iteration": i + 1,
            "query": current_query,
            "web": web_results,
            "kg": kg_results
        }
        all_iterations.append(iteration_data)
        
        if i < 3:
            decision = check_search_completeness(user_query, all_iterations)
            print(f"Reasoning: {decision.get('reasoning')}")
            if decision.get('is_complete'):
                break
            current_query = decision.get('next_query') or user_query
        else:
            print("Reached maximum iterations.")

    try:
        return synthesize_answer_with_llm(user_query, all_iterations)
    except Exception as exc:
        return f"Synthesis failed: {str(exc)}."
