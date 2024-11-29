from ollama import embeddings
import collections
import math
import json

from common import TimerLogger, chunkenize, llm, loadfiles, tokenize

DOCUMENT_FREQUENCY = "DOCUMENT_FREQUENCY"
INVERSE_DOCUMENT_FREQUENCY = "INVERSE_DOCUMENT_FREQUENCY"
TERM_FREQUENCY = "TERM_FREQUENCY"

preprocessing_timer = TimerLogger("Preprocessing")
preprocessing_timer.start()


x = [0,1,2,3,4,5,6]
print(x[:2:-1])

corpus_size = 0

EMBED_MODEL = 'nomic-embed-text'

document_vectors = []
chunk_store = {}

loaded_files = loadfiles()

def embed(text):
    embed_response = embeddings(model=EMBED_MODEL, prompt=text)
    return embed_response["embedding"]

for info in loaded_files:
    date = info["date"]
    #print(date)
    content = info["content"]
    corpus_size += len(content)

    chunks = chunkenize(content)

    for i, chunk in enumerate(chunks):
        id = f"{date}#{i}"

        vector = embed(chunk)

        document_vectors.append({"vector": vector, "id": id})
        chunk_store[id] = date + "\n" + chunk

preprocessing_timer.stop_and_log(corpus_size)

def cos_similarity(vector_a, vector_b):
    # if you use the same model, this shouldn't be a problem
    assert len(vector_a) == len(vector_a)

    similarity = 0

    sum_ab = 0
    sum_a2 = 0
    sum_b2 = 0
    # Inefficient? I don't fucking care
    for i, a in enumerate(vector_a):
        b = vector_b[i]
        sum_ab += a*b
        sum_a2 += a*a
        sum_b2 += b*b
    
    return sum_ab / (math.sqrt(sum_a2) * math.sqrt(sum_b2))

while True:
    query = input(">")
    query_timer = TimerLogger("Query")
    query_timer.start()

    embedded_query = embed(query)

    combined_scores = collections.Counter()

    for document_vector in document_vectors:
        score = cos_similarity(embedded_query, document_vector["vector"])
        combined_scores[document_vector["id"]] = score

    sorted_combined_scores = combined_scores.most_common()
    for chunk_id, score in sorted_combined_scores[:7]:
        print(score, chunk_id, chunk_store[chunk_id][:100].replace('\n', ''))
        #print(score, chunk_store[chunk_id])

    chunk_context = '\n\n'.join([chunk_store[i] for i,s in sorted_combined_scores[:2][::-1]])

    prompt = f"""
    Context:
    {chunk_context}

    Prompt:
    {query}

    Respond to the prompt using the information in the context. Do not explain anything, just reply in JSON format with the response and a step-by-step explanation. For example: {{"response": "Robs birthday is December 5th", "explanation": "The text mentions Robs birthday on December 5th"}}.
    """

    out = llm(prompt, True, False)
    # JSON isn't working perfectly. Rather than retrying, which could take fuckign forever, let's make the prompt better
    #obj = json.loads(out.strip())
    #print(obj["response"])

    query_timer.stop_and_log(corpus_size)
