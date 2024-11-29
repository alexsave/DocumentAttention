from ollama import generate
import time
import os
import re
import sys
from nltk.corpus import stopwords

LLAMA32 = "llama3.2"

stop = stopwords.words('english')

additional_terms = ['got', 'really', 'pretty', 'bit', 'didnt', 'get', 'also', 'like', 'went', 'go', 'im']
stop.extend(additional_terms)


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

    space_split = [re.sub(r"[.,’\-\?&;#!:\(\)''\"]", '', x) for x in space_split]
    space_split = [x for x in space_split if x not in stop ]

    return space_split

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



class TimerLogger:
    def __init__(self, label):
        self.label = label
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop_and_log(self, corpus_size):
        if self.start_time is None:
            raise ValueError("Timer was not started.")
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        print(f"{self.label} stats: {elapsed_time * 1000:.2f} milliseconds, {corpus_size} bytes, {elapsed_time / corpus_size:.6f} seconds/byte, {(elapsed_time * 1000 / corpus_size) * 1024 * 1024:.2f} milliseconds/MB")

