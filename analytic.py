import collections
import math
import json
import re  # Import regular expressions module

from common import TimerLogger, llm, loadfiles, tokenize, chunkenize  

INVERSE_DOCUMENT_FREQUENCY = "INVERSE_DOCUMENT_FREQUENCY"
TERM_FREQUENCY = "TERM_FREQUENCY"

# Preprocessing
preprocessing_timer = TimerLogger("Preprocessing")

corpus_size = 0
index = {}
chunk_store = {}

loaded_files = loadfiles()

for info in loaded_files:
    date = info["date"]
    content = info["content"]
    corpus_size += len(content)

    # Assuming 'chunkenize' splits content into chunks
    chunks = chunkenize(content)

    for i, chunk in enumerate(chunks):
        id = f"{date}#{i}"
        chunk_store[id] = f"{date}\n{chunk}"

        tokens = tokenize(chunk)
        document_len = len(tokens)
        for token in tokens:
            if token not in index:
                index[token] = {TERM_FREQUENCY: collections.Counter()}
            index[token][TERM_FREQUENCY][id] += 1.0 #/ document_len

chunk_count = len(chunk_store)
log_chunk_count = math.log(chunk_count)

for k, v in index.items():
    v[INVERSE_DOCUMENT_FREQUENCY] = log_chunk_count - math.log(len(v[TERM_FREQUENCY]))

preprocessing_timer.stop_and_log(corpus_size)
print(sum(index['jamie']['TERM_FREQUENCY'].values()))

# Sample Interaction
def sample_interaction():
    # Sample query
    query = input('user>')#"What are the top 5 most frequent tokens in the corpus?"

    # Prepare the prompt for the LLM
    prompt = f'''
You are an assistant that generates complete and functional Python code to solve the user's query using the available methods and data.

**Important Instructions:**
- Ensure that your code is fully implemented and does not contain any placeholders like `...` or comments such as `# implement this`.
- The code should be executable without any modifications.
- Do not add tests
- Do not add sample data
- Keep the code simple and concise, with no method definitions
- Do not initialize unnecessary variables
- Add detailed step-by-step comments
- Think carefully about types
- Do not print anything
- Save the final answer to `result`

**Available Data:**

- `index`: a dictionary where keys are tokens (strings), and values are dictionaries containing:
    - `{TERM_FREQUENCY}`: a `Counter` of chunk IDs to term frequency values.
    - `{INVERSE_DOCUMENT_FREQUENCY}`: a float representing the inverse document frequency of the token.
    - For example, `index`['home'][`{TERM_FREQUENCY}`] will return a `Counter` representing appearances of home

- `chunk_store`: a dictionary where keys are chunk IDs (strings), and values are chunks of text (strings).

Your task is to write Python code that uses the available data and methods to answer the following query:

\"\"\"{query}\"\"\"

Please reply with **only** the Python code, and ensure it is complete and ready to execute.
'''

    # Get Python code from the LLM
    code_response, stats = llm(prompt, log=True, user_log=True, format='')

    # Extract code between ```python and ```
    code_pattern = r'```python(.*?)```'
    matches = re.findall(code_pattern, code_response, re.DOTALL)
    if matches:
        code_to_execute = matches[0].strip()
    else:
        # If no ```python code block is found, try to extract code between ```
        code_pattern = r'```(.*?)```'
        matches = re.findall(code_pattern, code_response, re.DOTALL)
        if matches:
            code_to_execute = matches[0].strip()
        else:
            # No code fences found; assume the entire response is code
            code_to_execute = code_response.strip()

    # Evaluate the code
    try:
        exec_globals = {
            'index': index,
            'chunk_store': chunk_store,
            'tokenize': tokenize,
            'collections': collections,
            'math': math,
            'result': None  # Initialize result to None
        }
        exec(code_to_execute, exec_globals)
        result = exec_globals.get('result', None)
    except Exception as e:
        print(f"Error executing code: {e}")
        result = None

    # Check if result is obtained
    if result is None:
        print("No result was produced by the executed code.")
        return

    # Prepare the final prompt to get the answer
    answer_prompt = f'''
You are an assistant.

Here is the result of executing the code to solve the user's query:

{result}

Given the original question:

\"\"\"{query}\"\"\"

Please provide an answer based on the result.
'''

    # Get the final answer from the LLM
    final_answer, stats = llm(answer_prompt, log=True, user_log=False, format='')

    print("Final Answer:")
    print(final_answer)

# Run the sample interaction
if __name__ == "__main__":
    while True:
        sample_interaction()
