import fitz  # PyMuPDF
import base64
import requests
from PIL import Image
import json
import io
import os
import asyncio
import re
from tenacity import retry, stop_after_attempt, wait_exponential
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# ---------------------------
# CONFIG
# ---------------------------

# Ollama configuration
OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_VISION_MODEL = os.environ.get('OLLAMA_VISION_MODEL', 'llava')
OLLAMA_TEXT_MODEL = os.environ.get('OLLAMA_TEXT_MODEL', 'llama3')
# Load a small local embedding model (384 dimensions)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------------------
# HELPERS
# ---------------------------

def extract_json(text):
    if not text:
        return None

    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except:
        return None

async def generate_embedding(text):
    """Generates embedding for a single text string using local model."""
    # SentenceTransformer.encode is synchronous, but we can wrap it or just call it 
    # since it's relatively fast for single strings.
    # For better async performance with many strings, we could use a thread pool.
    emb = embedding_model.encode(text)
    return emb.tolist()

def clean_label(label):
    cleaned = ''.join(e for e in label if e.isalnum())
    return cleaned if cleaned else "Concept"

# ---------------------------
# PROMPT
# ---------------------------

PROMPT = """
### Role
You are a structural analyst and information extraction system. Your goal is to decompose document images into autonomous, semantically-rich "Knowledge Units" rather than atomic words.

### Task
Extract distinct Entities and the Relationships between them. A "Knowledge Unit" is a chunk of text that retains its full meaning when read in isolation.

### Extraction Guidelines
1.  **Prefer Chunks over Atoms:** Do not extract single words like "Gravity." Instead, extract the full descriptive unit: "Newton's Law of Universal Gravitation formula and its derivation."
2.  **Autonomous Entities:** Every entity must be "self-contained." This includes:
    * **Contextual Statements:** Complete sentences describing a specific fact or rule.
    * **Technical Blocks:** Entire formulas (including variable definitions), code snippets, or full tabular rows.
    * **Descriptive Clusters:** A heading combined with its immediate explanatory paragraph.
3.  **Dynamic Typing:** Since types are not fixed, assign a 'type' based on the chunk's function (e.g., "Theorem," "Statistical_Observation," "Procedural_Step," "System_Requirement").
4.  **Consistency:** Use these previously identified entities exactly if the content overlaps: {previous_entities}

### Relationship Mapping
Connect these chunks using active verbs that describe the flow of logic (e.g., "validates," "defines," "is_calculated_by," "provides_context_for").

### Output Format (Strict JSON)
{
  "entities": [
    {
      "name": "Full chunk text or concise summary of the block",
      "type": "Functional category (e.g., Technical_Specification, Definition_Block)",
      "metadata": "Briefly describe why this is a single unit"
    }
  ],
  "relationships": [
    {
      "source": "Exact 'name' of source entity",
      "relation": "Descriptive verb",
      "target": "Exact 'name' of target entity"
    }
  ]
}
"""

# ---------------------------
# PDF → Images
# ---------------------------

def pdf_to_images(pdf_content):
    """pdf_content can be a path or bytes"""
    if isinstance(pdf_content, bytes):
        doc = fitz.open(stream=pdf_content, filetype="pdf")
    else:
        doc = fitz.open(pdf_content)

    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)
    return images

    return images


# ---------------------------
# VECTOR INDEX SETUP
# ---------------------------

def setup_vector_index(tx):
    """Creates a vector index on the 'Concept' label for the 'embedding' property."""
    # Check if the index already exists to avoid errors
    result = tx.run("SHOW INDEXES YIELD name WHERE name = 'entity_embeddings' RETURN count(*) > 0 AS exists").single()
    if result and result["exists"]:
        return

    # Use the procedural call for compatibility with Neo4j 5.11+ (including 5.14.0)
    tx.run("CALL db.index.vector.createNodeIndex('entity_embeddings', 'Concept', 'embedding', 384, 'cosine')")


# ---------------------------
# IMAGE → LLM
# ---------------------------

async def extract_from_image(image, page_number, previous_entities):
    print(f"Processing page {page_number}... Image size: {image.size}, Mode: {image.mode}")
    
    formatted_prompt = PROMPT.replace(
        "{previous_entities}",
        json.dumps(previous_entities, indent=2) if previous_entities else "None"
    )

    # Convert PIL Image to base64 string
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_b64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    try:
        url = f"{OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": OLLAMA_VISION_MODEL,
            "prompt": formatted_prompt,
            "images": [img_b64],
            "stream": False
        }
        
        def fetch_ollama():
            res = requests.post(url, json=payload)
            res.raise_for_status()
            return res.json()
            
        # Use asyncio.to_thread to prevent blocking the event loop
        response_data = await asyncio.to_thread(fetch_ollama)
        text = response_data.get('response', '')
    except Exception as e:
        print(f"Ollama API Error for page {page_number}: {type(e).__name__}: {str(e)}")
        raise e
    
    try:
        data = extract_json(text)
        if not data:
             data = {"entities": [], "relationships": []}

        print(data['entities'])
    except:
        data = {"entities": [], "relationships": []}

    print(f"Finished page {page_number}")
    return {
        "page": page_number,
        "data": data
    }

# ---------------------------
# NEO4J UPLOAD FUNCTIONS
# ---------------------------

def create_nodes(tx, entities_metadata):
    """
    entities_metadata: dict mapping name -> {"type": typ, "page_numbers": set, "pdf_url": url, "embedding": list}
    """
    grouped = {}
    for name, meta in entities_metadata.items():
        label = clean_label(meta["type"])
        if label not in grouped:
            grouped[label] = []
        
        grouped[label].append({
            "name": name,
            "page_numbers": list(meta["page_numbers"]),
            "pdf_url": meta["pdf_url"],
            "embedding": meta.get("embedding")
        })

    for label, nodes in grouped.items():
        query = (
            f"UNWIND $nodes AS node "
            f"MERGE (n:`{label}` {{name: node.name}}) "
            f"SET n:Concept "
            f"SET n.page_numbers = node.page_numbers, "
            f"    n.pdf_url = node.pdf_url, "
            f"    n.embedding = node.embedding"
        )
        tx.run(query, nodes=nodes)

def create_edges(tx, relationships_with_meta):
    """
    relationships_with_meta: list of {"source": s, "relation": rel, "target": t, "page_number": p, "pdf_url": url}
    """
    grouped = {}
    for r in relationships_with_meta:
        rel_type = clean_label(r["relation"]).upper()
        if rel_type not in grouped:
            grouped[rel_type] = []
        grouped[rel_type].append(r)

    for rel_type, edges in grouped.items():
        query = (
            f"UNWIND $edges AS edge "
            f"MATCH (s {{name: edge.source}}) "
            f"MATCH (t {{name: edge.target}}) "
            f"MERGE (s)-[r:`{rel_type}`]->(t) "
            f"SET r.page_number = edge.page_number, "
            f"    r.pdf_url = edge.pdf_url"
        )
        tx.run(query, edges=edges)

# ---------------------------
# MAIN PIPELINE
# ---------------------------

async def process_pdf_to_neo4j(pdf_content, pdf_url, neo4j_config):
    """
    pdf_content: bytes or path
    pdf_url: string for traceability
    neo4j_config: {"uri": ..., "user": ..., "password": ...}
    """
    images = pdf_to_images(pdf_content)
    
    accumulated_entities = []
    entities_metadata = {} # name -> {"type": typ, "page_numbers": set, "pdf_url": url}
    relationships_with_meta = [] # list of dicts

    # Create debug directory
    os.makedirs("debug_images", exist_ok=True)
    
    for i, img in enumerate(images):
        page_num = i + 1
        
        # Save image for debugging
        debug_path = os.path.join("debug_images", f"page_{page_num}.png")
        img.save(debug_path)
        print(f"Saved debug image to {debug_path}")

        if len(accumulated_entities) > 100:
            given_entities = accumulated_entities[:100]
        else:
            given_entities = accumulated_entities
            
        result = await extract_from_image(img, page_num, given_entities)
        data = result["data"]

        # Track entities for LLM context and metadata
        for e in data.get("entities", []):
            name = e.get("name", "").strip()
            typ = e.get("type", "Unknown").strip()
            if name:
                if name not in entities_metadata:
                    entities_metadata[name] = {"type": typ, "page_numbers": set(), "pdf_url": pdf_url}
                    accumulated_entities.append(e)
                entities_metadata[name]["page_numbers"].add(page_num)

        # Track relationships with metadata
        for r in data.get("relationships", []):
            s = r.get("source", "").strip()
            rel = r.get("relation", "").strip()
            t = r.get("target", "").strip()
            if s and rel and t:
                relationships_with_meta.append({
                    "source": s,
                    "relation": rel,
                    "target": t,
                    "page_number": page_num,
                    "pdf_url": pdf_url
                })
                # Update page occurrence for existing entities found in relationships
                if s in entities_metadata:
                    entities_metadata[s]["page_numbers"].add(page_num)
                if t in entities_metadata:
                    entities_metadata[t]["page_numbers"].add(page_num)

    # Generate embeddings for all unique entities
    print(f"Generating embeddings for {len(entities_metadata)} entities...")
    for name in entities_metadata:
        entities_metadata[name]["embedding"] = await generate_embedding(name)

    # Upload to Neo4j
    with GraphDatabase.driver(neo4j_config["uri"], auth=(neo4j_config["user"], neo4j_config["password"])) as driver:
        with driver.session() as session:
            session.execute_write(setup_vector_index)
            session.execute_write(create_nodes, entities_metadata)
            session.execute_write(create_edges, relationships_with_meta)

    return {
        "status": "success",
        "entities_count": len(entities_metadata),
        "relationships_count": len(relationships_with_meta)
    }

# ---------------------------
# QUERY LOGIC
# ---------------------------

async def query_kg_by_vector(query_text, neo4j_config):
    """
    1. Extract entities from query
    2. Embed entities
    3. Search Neo4j for similar entities
    4. Fetch relationships
    """
    # 1. Extract entities from query using a simple LLM call
    extract_prompt = f"Extract important entities from this search query: '{query_text}'. Return them as a JSON list of strings: ['entity1', 'entity2']"
    
    try:
        url = f"{OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": OLLAMA_TEXT_MODEL,
            "prompt": extract_prompt,
            "stream": False
        }
        
        def fetch_ollama_text():
            res = requests.post(url, json=payload)
            res.raise_for_status()
            return res.json()
            
        response_data = await asyncio.to_thread(fetch_ollama_text)
        extracted_entities = extract_json(response_data.get('response', ''))
    except Exception as e:
        print(f"Ollama API Error for query extraction: {e}")
        extracted_entities = None
    if not extracted_entities or not isinstance(extracted_entities, list):
         # Fallback to the whole query if extraction fails
         extracted_entities = [query_text]

    all_results = {"entities": [], "relationships": []}
    seen_nodes = set()
    seen_rels = set()

    with GraphDatabase.driver(neo4j_config["uri"], auth=(neo4j_config["user"], neo4j_config["password"])) as driver:
        with driver.session() as session:
            for entity in extracted_entities:
                # 2. Embed
                emb = await generate_embedding(entity)
                
                # 3. Vector Search + Fetch Relationships
                query = """
                CALL db.index.vector.queryNodes('entity_embeddings', 5, $embedding)
                YIELD node, score
                MATCH (node)-[r]-(neighbor)
                RETURN node, r, neighbor, score
                LIMIT 20
                """
                results = session.run(query, embedding=emb)
                
                for record in results:
                    n = record["node"]
                    r = record["r"]
                    m = record["neighbor"]
                    
                    # Add node info
                    if n.element_id not in seen_nodes:
                        all_results["entities"].append({
                            "name": n["name"],
                            "type": list(n.labels)[0] if n.labels else "Unknown",
                            "score": record["score"]
                        })
                        seen_nodes.add(n.element_id)
                    
                    if m.element_id not in seen_nodes:
                        all_results["entities"].append({
                            "name": m["name"],
                            "type": list(m.labels)[0] if m.labels else "Unknown"
                        })
                        seen_nodes.add(m.element_id)

                    # Add relationship info
                    rel_id = r.element_id
                    if rel_id not in seen_rels:
                        all_results["relationships"].append({
                            "source": n["name"] if r.start_node == n else m["name"],
                            "relation": r.type,
                            "target": m["name"] if r.end_node == m else n["name"],
                            "page_number": r.get("page_number"),
                            "pdf_url": r.get("pdf_url")
                        })
                        seen_rels.add(rel_id)

    return all_results
