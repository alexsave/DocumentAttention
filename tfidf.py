import os
import time
import math

def tokenize(text):
    return []



start = time.time()

corpus_size = 0

chunk_size_bytes = 1024

sample_dir = 'sample_journals'

files_and_dirs = sorted(
    os.listdir(sample_dir),
    key=lambda x: os.path.getmtime(os.path.join(sample_dir, x))
)

index = {}

for file_path in files_and_dirs:
    with open(sample_dir + '/' + file_path, 'r') as file:
        date = os.path.basename(file_path).replace(".txt", "")
        print(date)
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
            print(tokenize(chunk))


            

after_preprocessing = time.time()
preprocessing_time = after_preprocessing - start

print(f"Preprocessing stats: {preprocessing_time*1000} milliseconds, {corpus_size} bytes, {preprocessing_time/corpus_size} seconds/byte, {preprocessing_time*1000/corpus_size*1024*1024} milliseconds/MB")

pre_query = time.time()
# do fun stuff
post_query = time.time()
query_time = pre_query - post_query