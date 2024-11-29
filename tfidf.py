import collections
import math
import json

from common import TimerLogger, chunkenize, llm, loadfiles, tokenize

INVERSE_DOCUMENT_FREQUENCY = "INVERSE_DOCUMENT_FREQUENCY"
TERM_FREQUENCY = "TERM_FREQUENCY"

preprocessing_timer = TimerLogger("Preprocessing")
preprocessing_timer.start()

corpus_size = 0

index = {}
chunk_store = {}

loaded_files = loadfiles()

for info in loaded_files:
    date = info["date"]
    #print(date)
    content = info["content"]
    corpus_size += len(content)

    chunks = chunkenize(content)

    for i, chunk in enumerate(chunks):
        id = f"{date}#{i}"

        # works a bit better with the date
        chunk_store[id] = date + "\n" + chunk

        tokens = tokenize(chunk)
        document_len = len(tokens)
        for token in tokens:
            if token not in index:
                index[token] = {TERM_FREQUENCY: collections.Counter()}
            index[token][TERM_FREQUENCY][id] += 1.0/document_len

chunk_count = len(chunk_store)
log_chunk_count = math.log(chunk_count)

for k,v in index.items():
    v[INVERSE_DOCUMENT_FREQUENCY] = log_chunk_count - math.log(len(v[TERM_FREQUENCY]))

total_term_frequencies = collections.Counter()
for token, data in index.items():
    total_term_frequencies[token] = sum(data[TERM_FREQUENCY].values())

# Find the most commonly used word
print(total_term_frequencies.most_common()[:10])

preprocessing_timer.stop_and_log(corpus_size)

while True:
    query = input(">")
    query_timer = TimerLogger("Query")
    query_timer.start()

    tokenized_query = tokenize(query)

    print(tokenized_query)

    combined_scores = collections.Counter()

    for token in tokenized_query:
        if token not in index:
            continue

        index_entry = index[token]
        
        inverse_document_frequency = index_entry[INVERSE_DOCUMENT_FREQUENCY]
        term_frequency = index_entry[TERM_FREQUENCY]

        scores = []

        # Calculate score for each chunk_id and chunk_store it in the list
        for chunk_id in term_frequency.keys():
            score = term_frequency[chunk_id] * inverse_document_frequency

            combined_scores[chunk_id] += score

            scores.append((score, chunk_id))
        
        # Sort the scores in descending order
        #sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
        
        # Print the top 7 scores and chunk_ids

    sorted_combined_scores = combined_scores.most_common()
    for chunk_id, score in sorted_combined_scores[:7]:
        print(score, chunk_id)
        #print(score, chunk_store[chunk_id])

    chunk_context = '\n\n'.join([chunk_store[i] for i,s in sorted_combined_scores[:2][::-1]])

    prompt = f"""
    Context:
    {chunk_context}

    Prompt:
    {query}

    Respond to the prompt using the information in the context. Do not explain anything, just reply in JSON format with the response and a step-by-step explanation. For example: {{"response": "Robs birthday is December 5th", "explanation": "The text mentions Robs birthday on December 5th"}}.
    """

    out = llm(prompt, True, False, format='json')
    obj = json.loads(out.strip())
    print(obj["response"])

    query_timer.stop_and_log(corpus_size)