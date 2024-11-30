import collections
import json
import pickle
import os
import hashlib
import tempfile

from common import ChatHistory, RetrievalHandler, TimerLogger, chunkenize, chunkenize_smalloverlap, llm, loadfiles, chunk_size_bytes

EMBED_MODEL = 'nomic-embed-text'

preprocessing_timer = TimerLogger("Preprocessing")

corpus_size = 0

RELATIONSHIPS_FILE = "relationships.pkl"

relationships_store = {}
chunk_store = {}

loaded_files = loadfiles()

# Compute a hash to verify the state of the input files
hash_input = pickle.dumps([chunk_size_bytes, EMBED_MODEL])
hash_value = hashlib.sha256(hash_input).hexdigest()

save_file = f"{hash_value[:7]}-{RELATIONSHIPS_FILE}"

# Function to save progress using pickle
def save_progress():
    with tempfile.NamedTemporaryFile('wb', delete=False) as temp_file:
        pickle.dump({"hash": hash_value, "relationships_store": relationships_store, }, temp_file)
        temp_file_path = temp_file.name
    os.replace(temp_file_path, save_file)

# Load relationships from file if they exist and match the hash
if os.path.exists(save_file):
    with open(save_file, 'rb') as f:
        try:
            saved_data = pickle.load(f)
            if saved_data.get("hash") == hash_value:
                relationships_store = saved_data["relationships_store"]
                #chunk_store = saved_data["chunk_store"]
                print("Loaded existing relationships from file.")
            else:
                print("Relationships file found but hash mismatch. Starting fresh.")
        except pickle.PickleError:
            print("Error decoding relationships file. Starting fresh.")
else:
    print("No existing relationships file found. Starting fresh.")

# Function to extract relationships in JSON format
def extract_relationships(chunk):
    prompt = f"""Extract all relationships between entities mentioned in the following text. For each relationship, provide it in JSON format with keys "subject", "predicate", and "object". Include all relevant relationships you can find. Do not include any text other than the JSON array of relationships.

Text:
{chunk}

Example Output:
[
  {{"subject": "Entity1", "predicate": "relation", "object": "Entity2"}},
  {{"subject": "Entity3", "predicate": "relation", "object": "Entity4"}}
]
"""

    response, stats = llm(prompt)
    # Parse the JSON response
    try:
        relationships = json.loads(response.strip())
        if not isinstance(relationships, list):
            relationships = []
    except json.JSONDecodeError:
        relationships = []
    print(relationships)
    return relationships

# Process chunks and extract relationships
chunks_processed = 0
for info in loaded_files:
    date = info["date"]
    content = info["content"]
    corpus_size += len(content)

    chunks = chunkenize_smalloverlap(content, 8192)

    for i, chunk in enumerate(chunks):
        id = f"{date}#{i}"
        print(id)
        #print(chunk)


        chunk_store[id] = date + "\n" + chunk

        # Skip if already processed
        if id in relationships_store:
            if chunks_processed % 5 == 0:
                chunks_processed += 1
            continue

        chunks_processed += 1
        relationships = extract_relationships(chunk)

        relationships_store[id] = {
            'date': date,
            'chunk': chunk,
            'relationships': relationships
        }

        # Save progress every 100 chunks
        if chunks_processed % 5 == 0:
            save_progress()
    # print(date)

# Save the final progress
save_progress()

preprocessing_timer.stop_and_log(corpus_size)

# Build an inverted index for quick lookup
inverted_index = {}
for doc_id, data in relationships_store.items():
    for rel in data['relationships']:
        subject = rel.get('subject', '').lower()
        predicate = rel.get('predicate', '').lower()
        obj = rel.get('object', '').lower() if rel.get('object', '') else ''
        key = (subject, predicate, obj)
        inverted_index.setdefault(key, set()).add(doc_id)

# Initialize chat history
chat_history = ChatHistory()

while True:
    query = input("user>")
    query_timer = TimerLogger("Query")

    if query == 'clear':
        chat_history.clear()
        print('system>cleared chat history')
        continue

    elif query == 'more':
        # Implement 'more' functionality if needed
        print('system>\'more\' functionality is not implemented.')
        continue

    else:
        chat_history.log_user(query)
        # Extract relationships from the query
        query_relationships = extract_relationships(query)

        # Find matching documents
        matched_docs = set()
        for rel in query_relationships:
            subject = rel.get('subject', '').lower()
            predicate = rel.get('predicate', '').lower()
            obj = rel.get('object', '').lower()
            key = (subject, predicate, obj)
            if key in inverted_index:
                matched_docs.update(inverted_index[key])

        # If no exact matches, try partial matches
        if not matched_docs:
            # Try matching on subject and predicate
            for rel in query_relationships:
                subject = rel.get('subject', '').lower()
                predicate = rel.get('predicate', '').lower()
                for stored_key in inverted_index.keys():
                    if stored_key[0] == subject and stored_key[1] == predicate:
                        matched_docs.update(inverted_index[stored_key])

            # Try matching on predicate and object
            if not matched_docs:
                for rel in query_relationships:
                    predicate = rel.get('predicate', '').lower()
                    obj = rel.get('object', '').lower()
                    for stored_key in inverted_index.keys():
                        if stored_key[1] == predicate and stored_key[2] == obj:
                            matched_docs.update(inverted_index[stored_key])

        # Retrieve and display the matched chunks
        if matched_docs:
            # Optionally, rank the matched documents
            doc_scores = {}
            for doc_id in matched_docs:
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1  # Simple count of matches

            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

            for doc_id, score in sorted_docs[:7]:  # Show top 7 matches
                data = relationships_store[doc_id]
                print(f"Document ID: {doc_id}")
                print(f"Date: {data['date']}")
                print(f"Score: {score}")
                print(f"Chunk: {data['chunk'][:500]}...")  # Print first 500 chars
                print("\n")
        else:
            print("No matching documents found.")

        # Optionally, generate a final response using the matched chunks
        if matched_docs:
            chunk_context = '\n\n'.join([chunk_store[doc_id] for doc_id, _ in sorted_docs[:7][::-1]])
            prompt = f"""Based on the following context, answer the user's question.

Context:
{chunk_context}

Question:
{query}

Provide a clear and concise answer in JSON format with a "response" key.

Example Output:
{{
  "response": "Your answer here."
}}
"""
            out, stats = llm(prompt, False, False, format='json', response_stream=True)
            try:
                obj = json.loads(out.strip())
                if 'response' in obj:
                    print(obj['response'])
                    chat_history.log_llm(obj['response'])
                else:
                    print("No 'response' field in the output.")
                    chat_history.log_llm("")
            except json.JSONDecodeError:
                print("Failed to parse LLM output as JSON.")
                chat_history.log_llm("")

    # query_timer.stop_and_log(corpus_size)
