import pickle
import tempfile
from ollama import generate
import time
import os
import re
import math
import sys
from nltk.corpus import stopwords

from ollama import embeddings
LLM_MODEL = "llama3.2"
EMBED_MODEL = 'nomic-embed-text'

chunk_size_bytes = 1024

# don't quote me on this
average_bytes_per_token = 3.5

stop = stopwords.words('english')

additional_terms = ['', 'got', 'really', 'pretty', 'bit', 'didnt', 'get', 'also', 'like', 'went', 'go', 'im']
stop.extend(additional_terms)



def final_prompt(context, query, use_history=None):
    return f"""
    Context:
    === start context ===
    {context}
    === end context ===

    {chat_history_block(use_history)}

    Prompt:
    {query}

    Respond to the prompt using the information in the context and chat history. Just reply in JSON format with a step-by-step explanation followed by a detailed and concise final response. Use just a single JSON object, e.g. {{"explanation": "1. [REASONING] 2. [REASONING] 3. [REASONING] ", "response": "[FINAL RESPONSE]"}}. Keep the "response" attribute a string.
    """
    #Respond to the prompt using the information in the context. Just reply in JSON format with a step-by-step explanation followed by a detailed and concise final response. Use just a single JSON object, e.g. {{"explanation": "1. The text mentions Robs birthday. 2. The text has the date 12/5. 3. ... ", "response": "Robs birthday is December 5th"}}.
    #Respond to the prompt using the information in the context. Do not explain anything, just reply in JSON format with the response and a step-by-step explanation. Just use a single JSON object, for example: {{"explanation": "1. The text mentions Robs birthday. 2. The text has the date 12/5. 3. ... ", "response": "Robs birthday is December 5th"}}.


def chat_history_block(history=None):
    return "Chat:\n=== start chat ===\n"+history.get_context()+"\n=== end chat ===" if history != None and len(history.get_context())>0 else ""


def chunkenize(content):
    chunks = []
    start_index = 0
    while start_index < len(content)-chunk_size_bytes/2:
        end_index = start_index + chunk_size_bytes
        chunks.append(content[start_index:end_index])
        start_index += int(chunk_size_bytes/2)
    return chunks


def llm(prompt, log=False, user_log=False, format='', response_stream=False):
    output = ""
    stats = {}
    if user_log:
        print(f"USER>{prompt}")
    if response_stream:
        # basically parse JSON in place
        response_end = False
        print(f"{LLM_MODEL}>", end='', flush=True)
        for part in generate(LLM_MODEL, prompt, stream=True, format=format):
            if 'prompt_eval_duration' in part:
                stats = part
            # this indicates that they've already printed the "response": part, and now we want the rest of the text
            # i assume the " is a token itself
            # let me set newline to check this real quick
            p = part['response']

            if response_end == False and ('"response": "' in output or '"response":"' in output):
                # bit hacky. not sure what to look for for the end of response
                # ah, so '."' is a token, as well as '}'
                # might be llama3.2 specific, but here goes
                # so what we're looking for is that response has start, but not finished yet
                if p == '."':
                    #print('first condition hit')
                    print('.', flush=True)
                    response_end = True
                elif p == '}':
                    #print('second condition hit')
                    response_end = True
                else:
                    print(p, end='', flush=True)
            output += p
        #print()

    elif log:
        print(f"{LLM_MODEL}>", end='')
        for part in generate(LLM_MODEL, prompt, stream=True, format=format):
            if 'prompt_eval_duration' in part:
                stats = part
            output += part['response']
            print(part['response'], end='', flush=True)
        print()

    else:
        stats = generate(LLM_MODEL, prompt, format=format)
        output = stats['response']

    return output, stats

def tokenize(text):
    space_split = [x.lower() for x in text.split()]

    space_split = [re.sub(r"[.,’\-\?&;#!:\(\)''\"]", '', x) for x in space_split]
    space_split = [x for x in space_split if x not in stop ]

    return space_split

def embed(text):
    embed_response = embeddings(model=EMBED_MODEL, prompt=text)
    return embed_response["embedding"]

def cos_similarity(vector_a, vector_b):
    # if you use the same model, this shouldn't be a problem
    assert len(vector_a) == len(vector_b)

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

def loadfiles():
    journal_dir = 'sample_journals'
    if len(sys.argv) > 1:
        journal_dir = sys.argv[1]
    files_and_dirs = sorted(
        os.listdir(journal_dir),
        key=lambda x: os.path.getmtime(os.path.join(journal_dir, x))
    )
    pattern = re.compile("[12].*")
    files_and_dirs = [ x for x in files_and_dirs if re.match(pattern, x)]

    result = []
    for file_path in files_and_dirs:
        with open(journal_dir + '/' + file_path, 'r') as file:
            date = os.path.basename(file_path).replace(".txt", "")
            content = file.read()
            result.append({"date": date, "content": content})
    
    return result

# Function to save progress using pickle
#data is a dict of whatever
def save_progress(save_file, data):
    with tempfile.NamedTemporaryFile('wb', delete=False) as temp_file:
        pickle.dump(data, temp_file)
        temp_file_path = temp_file.name
    os.replace(temp_file_path, save_file)

class TimerLogger:
    def __init__(self, label):
        self.label = label
        self.start_time = time.time()

    def stop_and_log(self, corpus_size):
        if self.start_time is None:
            raise ValueError("Timer was not started.")
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        print(f"{self.label} stats: {elapsed_time * 1000:.2f} milliseconds, {corpus_size} bytes, {(elapsed_time * 1000 / corpus_size) * 1024 * 1024:.2f} milliseconds/MB, {(elapsed_time / corpus_size) * 1024*1024:.4f} seconds/MB")


def expand(query, type='tfidf', history=None):
    prompt = ""
    if type == 'tfidf':
        prompt = f"""Expand the following query using related terms, synonyms, alternate phrasings, and contextual keywords, considering the context provided in the chat history to improve relevance for searching in a TFIDF index. Output only the expanded query as a list of terms and phrases, without any explanation or additional context.

{chat_history_block(history)}

Query: {query}"""
    else:
        prompt = f"""
Expand the following query using related terms, synonyms, semantic variations, and contextual keywords, considering the context provided in the chat history to enhance relevance for searching in a vector database. Output only the expanded query as a list of terms and phrases, without any explanation or additional context.

{chat_history_block(history)}

Query: {query}"""

    output,_= llm(prompt, True, True)
    return output


# assuming this handles garbage collection automatically
# this has a lot of queries, could probably more more stuff to this and avoid params
class RetrievalHandler:
    def __init__(self, query, full_scores, chunk_store, page_size=20, history=None):
        self.query = query
        self.full_scores = full_scores
        self.page_size = page_size
        self.chunk_store = chunk_store
        self.history = history
        # essentially pagination
        self.start = 0

    def has_more(self):
        return self.start < len(self.full_scores)

    # returns another prompt
    def __get_next_page(self):
        # we can probably put the whole thing together here
        res = self.full_scores[self.start:(self.start+self.page_size)]
        self.start += self.page_size
        return res

    def build_prompt(self):
        scores = []
        if self.has_more():
            scores = self.__get_next_page()
        chunk_context = '\n\n'.join([self.chunk_store[i] for i,_ in scores[::-1]])
        prompt = final_prompt(chunk_context, self.query, use_history=self.history)
        print(prompt)
        return prompt

class ChatHistory:
    def __init__(self):
        self.history = []

    def log_user(self, text):
        self.history.append({"role":"user", "text":text})

    def log_llm(self, text):
        self.history.append({"role":"llm", "text":text})

    def get_context(self):
        return '\n'.join([f"{h["role"]}: {h["text"]}" for h in self.history[-7:-1]])

    def clear(self):
        # hope this doesn't cause problems
        self.__init__()
