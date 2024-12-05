import collections
import math
import json
import os
import pickle
import hashlib

from common import EMBED_MODEL, RetrievalHandler, TimerLogger, chunkenize, chunkenize_smalloverlap, llm, loadfiles, tokenize, chunk_size_bytes

preprocessing_timer = TimerLogger("Preprocessing")

corpus_size = 0

# Load the existing sentiment data from the previous sentiment file
SENTIMENT_FILE = "sentiment.pkl"

sentiment_store = {}
word_sentiment = collections.Counter()
word_counts = collections.Counter()

loaded_files = loadfiles()

# Compute a hash to verify the state of the input files
hash_input = pickle.dumps([chunk_size_bytes, EMBED_MODEL])
hash_value = hashlib.sha256(hash_input).hexdigest()

# Use the same sentiment file as before
save_file = f"{hash_value[:7]}-{SENTIMENT_FILE}"

# not exactly stopwords, but not what I'm looking for and not relevant to specific thing
ignore_words = ['good', 'day', 'one', 'today', 'back', 'much', 'wasnt', 'even', 'know', 'actually', 'would', 'took', 'dont', 'time', 'still', 'place', 'year', 'going', 'thats', 'could', 'well', 'around']
#ignore_words = []

# Load sentiment data from the existing sentiment file
if os.path.exists(save_file):
    with open(save_file, 'rb') as f:
        try:
            saved_data = pickle.load(f)
            if saved_data.get("hash") == hash_value:
                sentiment_store = saved_data["sentiment_store"]
                print("Loaded existing sentiment data from file.")
            else:
                print("Sentiment file found but hash mismatch. Starting fresh.")
        except pickle.PickleError:
            print("Error decoding sentiment file. Starting fresh.")
else:
    print("No existing sentiment file found. Please run the sentiment analysis code first.")
    exit()

# Process chunks to build word sentiment mappings

# Reconstruct chunks from loaded_files and process them
for info in loaded_files:
    date_str = info["date"]
    print(f"Original date string: {date_str}")
    content = info["content"]
    corpus_size += len(content)

    # Reconstruct the chunks
    chunks = chunkenize_smalloverlap(content, 8192)

    for i, chunk in enumerate(chunks):
        id = f"{date_str}#{i}"

        # Check if this chunk ID is in sentiment_store
        if id in sentiment_store:
            sentiment_score = sentiment_store[id]['sentiment_score']

            # Normalize sentiment score
            if sentiment_score is not None:
                sentiment_normalized = (sentiment_score - 50) / 50.0  # Normalize between -1 and 1
            else:
                sentiment_normalized = 0  # Treat as neutral if sentiment_score is None

            # Tokenize the chunk
            tokens = tokenize(chunk)
            unique_tokens = set(tokens)  # Avoid counting duplicate words in the same chunk

            # eh, if a word is mentioned a lot in a really happy entry, it's a pretty strong signal
            #for token in tokens:
            for token in tokens:
                if token == 'wrestled':
                    print(sentiment_normalized)

                word_sentiment[token] += sentiment_normalized
                word_counts[token] += 1

# After processing, calculate average sentiment per word
word_avg_sentiment = {}
for word in word_sentiment:
    # if a word is only mentioned once in a really negative entry, it gets ranked low. and vice versa
    if word not in ignore_words and word_counts[word] > 10:
        # maybe don't normalize. This kinda gets around the word count issue
        word_avg_sentiment[word] = word_sentiment[word] / word_counts[word]

# Now, sort words by average sentiment
sorted_words_positive = sorted(word_avg_sentiment.items(), key=lambda x: x[1], reverse=True)
sorted_words_negative = sorted(word_avg_sentiment.items(), key=lambda x: x[1])

print("Top words correlated with happiness:")
for word, avg_sent in sorted_words_positive[:40]:
    print(f"{word}: {avg_sent:.3f}")

print("\nTop words correlated with unhappiness:")
for word, avg_sent in sorted_words_negative[:40]:
    print(f"{word}: {avg_sent:.3f}")
