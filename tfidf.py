import os
import collections
import time
import math
import sys
import re
import json
from ollama import generate

LLAMA32 = "llama3.2"

def llm(prompt, log=False, user_log=False):
    output = ""
    if user_log:
        print(f"USER>{prompt}")
    if log:
        print(f"{LLAMA32}>", end='')
    for part in generate(LLAMA32, prompt, stream=True):
        output += part['response']
        if log:
            print(part['response'], end='', flush=True)
    if log:
        print()
    return output

def tokenize(text):
    space_split = [x.lower() for x in text.split()]
    #print(space_split)


    #exit(-1)
    return space_split


DOCUMENT_FREQUENCY = "DOCUMENT_FREQUENCY"
INVERSE_DOCUMENT_FREQUENCY = "INVERSE_DOCUMENT_FREQUENCY"
TERM_FREQUENCY = "TERM_FREQUENCY"

start = time.time()

corpus_size = 0

chunk_size_bytes = 1024

journal_dir = 'sample_journals'
if sys.argv[1]:
    journal_dir = sys.argv[1]

files_and_dirs = sorted(
    os.listdir(journal_dir),
    key=lambda x: os.path.getmtime(os.path.join(journal_dir, x))
)

index = {}
chunk_store = {}

pattern = re.compile("[12].*")
files_and_dirs = [ x for x in files_and_dirs if re.match(pattern, x)]
for file_path in files_and_dirs:
    with open(journal_dir + '/' + file_path, 'r') as file:
        date = os.path.basename(file_path).replace(".txt", "")
        #print(date)
        content = file.read()
        corpus_size += len(content)


        chunks = []
        #for start_index in range(0, )
        start_index = 0
        while start_index < len(content)-chunk_size_bytes/2:
            end_index = start_index + chunk_size_bytes
            chunks.append(content[start_index:end_index])
            start_index += int(chunk_size_bytes/2)


        for i, chunk in enumerate(chunks):
            id = f"{date}#{i}"

            # works a bit better with the date
            chunk_store[id] = date + "\n" + chunk

            tokens = tokenize(chunk)
            document_len = len(tokens)
            for token in tokens:
                if token not in index:
                    index[token] = {TERM_FREQUENCY: collections.Counter()}#, "document_frequency": 0}
                #if id not in index[token]["matches"]:
                index[token][TERM_FREQUENCY][id] += 1.0/document_len

chunk_count = len(chunk_store)
log_chunk_count = math.log(chunk_count)

for k,v in index.items():
    v[INVERSE_DOCUMENT_FREQUENCY] = log_chunk_count - math.log(len(v[TERM_FREQUENCY]))


after_preprocessing = time.time()
preprocessing_time = after_preprocessing - start

print(f"Preprocessing stats: {preprocessing_time*1000} milliseconds, {corpus_size} bytes, {preprocessing_time/corpus_size} seconds/byte, {preprocessing_time*1000/corpus_size*1024*1024} milliseconds/MB")

while True:
    query = input(">")
    pre_query = time.time()

    tokenized_query = tokenize(query)

    print(tokenized_query)

    combined_scores = collections.Counter()

    for token in tokenized_query:
        if token not in index:
            continue

        index_entry = index[token]
        #document_frequency = index_entry[DOCUMENT_FREQUENCY]
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


    chunk_context = '\n\n'.join([chunk_store[i] for i,s in sorted_combined_scores[:7:-1]])

    prompt = f"""
    Context:
    {chunk_context}

    Prompt:
    {query}

    Respond to the prompt using the information in the context. Do not explain anything, just reply in JSON format with the response and a step-by-step explanation. For example: {{"response": "Robs birthday is December 5th", "explanation": "The text mentions Robs birthday on December 5th"}}.
    """

    out = llm(prompt, True, False)
    obj = json.loads(out.strip())
    print(obj["response]"])

    for part in generate(LLAMA32, prompt, stream=True):
        print(part['response'], end='', flush=True)
    
    
    post_query = time.time()
    query_time = post_query - pre_query
    print(f"Query stats: {query_time*1000} milliseconds, {corpus_size} bytes, {query_time/corpus_size} seconds/byte, {query_time*1000/corpus_size*1024*1024} milliseconds/MB")