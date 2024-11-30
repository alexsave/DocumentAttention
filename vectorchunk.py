import collections
import pickle
import os
import hashlib
import tempfile

from common import TimerLogger, chunkenize, cos_similarity, embed, final_prompt, llm, loadfiles, chunk_size_bytes

DOCUMENT_FREQUENCY = "DOCUMENT_FREQUENCY"
INVERSE_DOCUMENT_FREQUENCY = "INVERSE_DOCUMENT_FREQUENCY"
TERM_FREQUENCY = "TERM_FREQUENCY"

preprocessing_timer = TimerLogger("Preprocessing")
preprocessing_timer.start()

corpus_size = 0

EMBED_MODEL = 'nomic-embed-text'

EMBEDDINGS_FILE = "embeddings.pkl"

document_vectors = {}
# starting to think it might not be a good idea to store chunks, as we basically duplicate everything
# but then again, the vectors take up WAY more space
chunk_store = {}

loaded_files = loadfiles()

# Compute a hash to verify the state of the input files
hash_input = pickle.dumps([chunk_size_bytes, EMBED_MODEL])
hash_value = hashlib.sha256(hash_input).hexdigest()

save_file = f"{hash_value[:7]}-{EMBEDDINGS_FILE}"

# Function to save progress using pickle
def save_progress():
    with tempfile.NamedTemporaryFile('wb', delete=False) as temp_file:
        pickle.dump({"hash": hash_value, "document_vectors": document_vectors}, temp_file)
        temp_file_path = temp_file.name
    os.replace(temp_file_path, save_file)

# Load embeddings from file if they exist and match the hash
if os.path.exists(save_file):
    with open(save_file, 'rb') as f:
        try:
            saved_data = pickle.load(f)
            if saved_data.get("hash") == hash_value:
                document_vectors = saved_data["document_vectors"]
                print("Loaded existing embeddings from file.")
            else:
                print("Embeddings file found but hash mismatch. Starting fresh.")
        except pickle.PickleError:
            print("Error decoding embeddings file. Starting fresh.")
else:
    print("No existing embeddings file found. Starting fresh.")

# Embed chunks and save to file after each document is processed
chunks_processed = 0
for info in loaded_files:
    date = info["date"]
    content = info["content"]
    corpus_size += len(content)

    chunks = chunkenize(content)

    for i, chunk in enumerate(chunks):
        id = f"{date}#{i}"

        chunk_store[id] = date + "\n" + chunk
        # Skip if already embedded
        if id in document_vectors:
            # little trick so that it doesn't save repeatedly when we've already processed chunks in json, but haven't seen them here
            if chunks_processed % 100 == 0:
                chunks_processed += 1
            continue

        chunks_processed += 1
        vector = embed(chunk)

        document_vectors[id] = vector

        # Save progress to file using a temporary file to avoid corruption. But don't do it too much, it slows down pre-processing
        if chunks_processed % 100 == 0:
            save_progress()
    print(date)

# one last time
save_progress()

preprocessing_timer.stop_and_log(corpus_size)


while True:
    query = input(">")
    query_timer = TimerLogger("Query")
    query_timer.start()

    embedded_query = embed(query)

    combined_scores = collections.Counter()

    for k,v in document_vectors.items():
        score = cos_similarity(embedded_query, v)
        combined_scores[k] = score

    sorted_combined_scores = combined_scores.most_common()
    for chunk_id, score in sorted_combined_scores[:7]:
        print(score, chunk_id, chunk_store[chunk_id][:100].replace('\n', ''))
        #print(score, chunk_store[chunk_id])

    chunk_context = '\n\n'.join([chunk_store[i] for i,s in sorted_combined_scores[:7][::-1]])

    prompt = final_prompt(chunk_context, query)

    out = llm(prompt, True, True, format='json')
    # JSON isn't working perfectly. Rather than retrying, which could take fuckign forever, let's make the prompt better
    #obj = json.loads(out.strip())
    #print(obj["response"])

    query_timer.stop_and_log(corpus_size)
