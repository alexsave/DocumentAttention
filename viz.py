import collections
import json
import pickle
import os
import hashlib
import tempfile
import networkx as nx
from pyvis.network import Network  # Import PyVis

from common import ChatHistory, RetrievalHandler, TimerLogger, chunkenize, llm, loadfiles, chunk_size_bytes

EMBED_MODEL = 'nomic-embed-text'

preprocessing_timer = TimerLogger("Preprocessing")

corpus_size = 0

RELATIONSHIPS_FILE = "relationships.pkl"

relationships_store = {}
chunk_store = {}
G = nx.DiGraph()  # Initialize a directed graph

loaded_files = loadfiles()

# Compute a hash to verify the state of the input files
hash_input = pickle.dumps([chunk_size_bytes, EMBED_MODEL])
hash_value = hashlib.sha256(hash_input).hexdigest()

save_file = f"{hash_value[:7]}-{RELATIONSHIPS_FILE}"

# Function to save progress using pickle
def save_progress():
    with tempfile.NamedTemporaryFile('wb', delete=False) as temp_file:
        pickle.dump({
            "hash": hash_value,
            "relationships_store": relationships_store,
            "chunk_store": chunk_store
        }, temp_file)
        temp_file_path = temp_file.name
    os.replace(temp_file_path, save_file)

# Load relationships from file if they exist and match the hash
relationships_loaded = False  # Flag to check if relationships were loaded from disk
if os.path.exists(save_file):
    with open(save_file, 'rb') as f:
        try:
            saved_data = pickle.load(f)
            if saved_data.get("hash") == hash_value:
                relationships_store = saved_data["relationships_store"]
                chunk_store = saved_data["chunk_store"]
                print("Loaded existing relationships from file.")
                relationships_loaded = True
            else:
                print("Relationships file found but hash mismatch. Starting fresh.")
        except pickle.PickleError:
            print("Error decoding relationships file. Starting fresh.")
else:
    print("No existing relationships file found. Starting fresh.")

# Function to extract relationships in JSON format
def extract_relationships(chunk):
    prompt = f"""You are to extract all relationships between entities mentioned in the following text. For each relationship, provide it as an object with keys "subject", "predicate", and "object". Return a JSON array of these relationship objects. Do not include any text other than the JSON array.

Text:
\"\"\"
{chunk}
\"\"\"

Example Output:
[
  {{"subject": "Entity1", "predicate": "relation", "object": "Entity2"}},
  {{"subject": "Entity3", "predicate": "relation", "object": "Entity4"}}
]
"""
    response, stats = llm(prompt, temperature=0.1)  # Set low temperature

    # Parse the JSON response
    try:
        relationships = json.loads(response.strip())
        if not isinstance(relationships, list):
            relationships = []
    except json.JSONDecodeError as e:
        print("JSONDecodeError:", e)
        relationships = []
    return relationships

# Process chunks and extract relationships
chunks_processed = 0
if not relationships_loaded:
    for info in loaded_files:
        date = info["date"]
        content = info["content"]
        corpus_size += len(content)

        chunks = chunkenize(content)

        for i, chunk in enumerate(chunks):
            id = f"{date}#{i}"

            chunk_store[id] = date + "\n" + chunk

            # Skip if already processed
            if id in relationships_store:
                if chunks_processed % 100 == 0:
                    chunks_processed += 1
                continue

            chunks_processed += 1
            relationships = extract_relationships(chunk)

            relationships_store[id] = {
                'date': date,
                'chunk': chunk,
                'relationships': relationships
            }

            # Build the graph
            for rel in relationships:
                subject = rel.get('subject', '').strip()
                predicate = rel.get('predicate', '').strip()
                obj = rel.get('object', '').strip()
                if subject and predicate and obj:
                    G.add_edge(subject, obj, label=predicate)

            # Save progress every 100 chunks
            if chunks_processed % 100 == 0:
                save_progress()
    # Save the final progress
    save_progress()
else:
    # Build the graph from the loaded relationships_store
    for data in relationships_store.values():
        relationships = data['relationships']
        for rel in relationships:
            subject = rel.get('subject', '').strip() if rel.get('subject','') else ''
            predicate = rel.get('predicate', '').strip() if rel.get('predicate','') else ''
            obj = rel.get('object', '').strip() if rel.get('object','') else ''
            if subject and predicate and obj:
                G.add_edge(subject, obj, label=predicate)

#preprocessing_timer.stop_and_log(corpus_size)

# Visualize the graph with physics using PyVis
def visualize_graph_with_pyvis(G, min_degree=2):
    if G.number_of_nodes() == 0:
        print("Graph is empty. Nothing to visualize.")
        return

    # Filter nodes with degree >= min_degree
    nodes_with_min_degree = [node for node in G.nodes() if G.degree(node) >= min_degree]

    # Create subgraph with nodes that have at least min_degree connections
    subgraph = G.subgraph(nodes_with_min_degree).copy()

    if subgraph.number_of_nodes() == 0:
        print(f"No nodes with at least {min_degree} connections.")
        return

    net = Network(height='750px', width='100%', notebook=False, directed=True)
    net.barnes_hut()  # Enable physics simulation

    # Add nodes and edges to the PyVis network
    for node in subgraph.nodes():
        net.add_node(node, label=node, title=node)

    for source, target, data in subgraph.edges(data=True):
        predicate = data.get('label', '')
        net.add_edge(source, target, label=predicate, title=predicate)

    # Customize physics settings (optional)
    physics_options = {
        'enabled': True,
        'barnesHut': {
            'gravitationalConstant': -80000,
            'centralGravity': 0.9,
            'springLength': 95,
            'springConstant': 0.04,
            'damping': 0.09,
            'avoidOverlap': 0
        },
        'minVelocity': 0.75
    }
    net.set_options("""
    var options = {
      "physics": %s
    }
    """ % json.dumps(physics_options))

    #net.show_buttons(filter_=['physics'])  # Show physics configuration UI
    net.show('graph.html', notebook=False)  # Save and open the graph in a web browser

# Call the PyVis visualization function
visualize_graph_with_pyvis(G, min_degree=1)