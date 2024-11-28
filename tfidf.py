import os
import collections
import time
import math

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

sample_dir = 'sample_journals'

files_and_dirs = sorted(
    os.listdir(sample_dir),
    key=lambda x: os.path.getmtime(os.path.join(sample_dir, x))
)

index = {}
chunk_store = {}

for file_path in files_and_dirs:
    with open(sample_dir + '/' + file_path, 'r') as file:
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

            chunk_store[id] = chunk

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
        #for score, chunk_id in sorted_scores[:1]:
            #print(score, chunk_store[chunk_id])

    sorted_combined_scores = combined_scores.most_common()
    for chunk_id, score in sorted_combined_scores[:7]:
        print(score, chunk_store[chunk_id])
        #print(score, chunk_id)

    
    
    # do fun stuff
    post_query = time.time()
    query_time = post_query - pre_query
    print(f"Query stats: {query_time*1000} milliseconds, {corpus_size} bytes, {query_time/corpus_size} seconds/byte, {query_time*1000/corpus_size*1024*1024} milliseconds/MB")