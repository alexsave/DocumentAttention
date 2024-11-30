import collections
import pickle
import os
import hashlib

from common import EMBED_MODEL, TimerLogger, chunkenize, cos_similarity, embed, final_prompt, llm, loadfiles, chunk_size_bytes, save_progress

DOCUMENT_FREQUENCY = "DOCUMENT_FREQUENCY"
INVERSE_DOCUMENT_FREQUENCY = "INVERSE_DOCUMENT_FREQUENCY"
TERM_FREQUENCY = "TERM_FREQUENCY"

preprocessing_timer = TimerLogger("Preprocessing")
preprocessing_timer.start()

corpus_size = 0

ATTENTION_FILE = "attention.pkl"
#Cool idea in theory but very very very slow
DIMENSION_PROMPTS = {
  "Summary": {
    "document_prompt": "Summarize the main story or sequence of events.",
    "query_prompt": "Summarize the main intent or request expressed in the text."
  },
  "People": {
    "document_prompt": "Extract all key people mentioned in the text.",
    "query_prompt": "Identify any people mentioned or referred to in the text."
  },
  "Places": {
    "document_prompt": "Extract all key places mentioned in the text.",
    "query_prompt": "Identify any places mentioned or referred to in the text."
  },
  "Organizations": {
    "document_prompt": "Extract all key organizations mentioned in the text.",
    "query_prompt": "Identify any organizations mentioned or referred to in the text."
  },
  "Objects": {
    "document_prompt": "Extract all key objects mentioned in the text.",
    "query_prompt": "Identify any objects or items mentioned or referred to in the text."
  },
  "Actions/Events": {
    "document_prompt": "Identify all key actions or events described, focusing on central activities.",
    "query_prompt": "Identify the key actions or events the text is interested in or is requesting information about."
  },
  "Concepts/Themes": {
    "document_prompt": "Extract and explain key concepts, themes, or ideas that are central to the text but may not be tied to concrete entities.",
    "query_prompt": "Extract and explain key concepts, themes, or ideas that are central to the text."
  },
  "Emotional Tone/Sentiment": {
    "document_prompt": "Analyze the overall emotional tone of the text and any sentiments expressed towards specific entities (people, organizations, etc.).",
    "query_prompt": "Analyze the overall emotional tone or sentiment of the text, and any sentiments expressed towards specific entities."
  },
  "Relationships": {
    "document_prompt": "Identify and describe relationships between key people, places, organizations, and objects mentioned in the text.",
    "query_prompt": "Identify and describe any relationships between people, places, organizations, or objects mentioned in the text."
  },
  "Cause-and-effect": {
    "document_prompt": "Extract any cause-and-effect relationships described.",
    "query_prompt": "Identify any cause-and-effect relationships implied or stated in the text."
  },
  "Motivations": {
    "document_prompt": "Identify reasons or goals behind actions or feelings (motivations).",
    "query_prompt": "Identify reasons or motivations behind the text."
  },
  "Time References": {
    "document_prompt": "Extract any explicit or implicit references to time (dates, seasons, periods).",
    "query_prompt": "Extract any explicit or implicit references to time (dates, seasons, periods) in the text."
  },
  "Significant Locations": {
    "document_prompt": "Identify significant locations and explain their contextual importance.",
    "query_prompt": "Identify any significant locations mentioned in the text and explain their relevance."
  },
  "Recurring Themes/Behaviors": {
    "document_prompt": "Analyze the text for any recurring behaviors, themes, or trends.",
    "query_prompt": "Identify any recurring behaviors, themes, or trends the user is interested in based on the text."
  },
  "Writing Style/Structure": {
    "document_prompt": "Describe the writing style, tone, and notable structural elements of the text (e.g., narrative style, use of dialogue).",
    "query_prompt": "Analyze the language style and tone of the text (e.g., formal, informal, urgent, inquisitive)."
  },
  "Cultural/Historical Context": {
    "document_prompt": "Identify any cultural, historical, or societal contexts referenced and explain their significance to the text.",
    "query_prompt": "Identify any cultural, historical, or societal contexts referenced in the text and explain their significance."
  },
  "Contradictions/Inconsistencies": {
    "document_prompt": "Identify any contradictions or inconsistencies within the text and discuss their potential impact.",
    "query_prompt": "Identify any contradictions or inconsistencies within the text and discuss their potential impact on understanding the request."
  },
  "Intended Audience/Purpose": {
    "document_prompt": "Analyze the intended audience and purpose of the text, including any calls to action or persuasive elements.",
    "query_prompt": "Analyze the intended purpose of the text, including any specific requests or desired outcomes."
  },
  "Ethical/Moral Dilemmas": {
    "document_prompt": "Identify any ethical or moral dilemmas presented and discuss their significance.",
    "query_prompt": "Identify any ethical or moral issues raised in the text and discuss their significance."
  },
  "Literary Devices/Techniques": {
    "document_prompt": "Identify any rhetorical devices or literary techniques used (e.g., metaphors, similes, analogies) and discuss their effect.",
    "query_prompt": "Identify any rhetorical devices or expressions used in the text and discuss their effect."
  }
}



# will possibly replace document_vectors
dimension_vectors = {}
# starting to think it might not be a good idea to store chunks, as we basically duplicate everything
# but then again, the vectors take up WAY more space
chunk_store = {}

loaded_files = loadfiles()

# Compute a hash to verify the state of the input files
hash_input = pickle.dumps([chunk_size_bytes, EMBED_MODEL, DIMENSION_PROMPTS])
hash_value = hashlib.sha256(hash_input).hexdigest()

save_file = f"{hash_value[:7]}-{ATTENTION_FILE}"

def extract_metadata(chunk, type='document'):
    metadata = {}
    suffix = " Do not explain anything or repeat the question, just answer. The response will be put into a vector db. Keep the response to a concise sentence."
    for key, prompts in DIMENSION_PROMPTS.items():
        full_prompt = f"{prompts[("document_prompt" if type=='document' else "query_prompt")]}{suffix}\n\nText:\n{chunk}"
        response, stats = llm(full_prompt)
        # should we keep original response rather than just embed?
        metadata[key] = embed(response.strip())

    # this just takes the chunk and embeds it. could be useful, we'll see. 
    #metadata["raw"] = embed(chunk)
    return metadata

# Load embeddings from file if they exist and match the hash
if os.path.exists(save_file):
    with open(save_file, 'rb') as f:
        try:
            saved_data = pickle.load(f)
            if saved_data.get("hash") == hash_value:
                dimension_vectors = saved_data["dimension_vectors"]
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
        print(id)

        chunk_store[id] = date + "\n" + chunk
        # Skip if already embedded
        if id in dimension_vectors:
            # little trick so that it doesn't save repeatedly when we've already processed chunks in json, but haven't seen them here
            if chunks_processed % 5 == 0:
                chunks_processed += 1
            continue

        chunks_processed += 1
        metadata = extract_metadata(chunk)

        dimension_vectors[id] = metadata

        # Save progress to file using a temporary file to avoid corruption. But don't do it too much, it slows down pre-processing
        # save more frequently cuz each chunk is slow af
        if chunks_processed % 5 == 0:
            save_progress(save_file, {"hash": hash_value, "dimension_vectors": dimension_vectors})
    print(date)

# one last time
save_progress(save_file, {"hash": hash_value, "dimension_vectors": dimension_vectors})

preprocessing_timer.stop_and_log(corpus_size)

while True:
    query = input(">")
    query_timer = TimerLogger("Query")
    query_timer.start()

    embedded_query = embed(query)

    # we need to get query metadata
    query_heads = extract_metadata(query, type='query')

    # then we need to ask it one more time to get weights for each dimension, but for now we'll set all weights to 1

    combined_scores = collections.Counter()

    for k,v in dimension_vectors.items():
        score = 0
        for dim in DIMENSION_PROMPTS.keys():
            sim = cos_similarity(query_heads[dim], v[dim])
            print(k, query, dim, sim)
            score += sim

        combined_scores[k] = score
        print(k, query, score)

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
