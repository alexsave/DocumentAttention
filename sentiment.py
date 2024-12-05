import collections
import json
import pickle
import os
import hashlib
import tempfile

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandas as pd
from datetime import datetime
from dateutil import parser

from common import ChatHistory, RetrievalHandler, TimerLogger, chunkenize, chunkenize_smalloverlap, llm, loadfiles, chunk_size_bytes

EMBED_MODEL = 'nomic-embed-text'

preprocessing_timer = TimerLogger("Preprocessing")

corpus_size = 0

SENTIMENT_FILE = "sentiment.pkl"

sentiment_store = {}
chunk_store = {}

loaded_files = loadfiles()

show_year = False

# Compute a hash to verify the state of the input files
hash_input = pickle.dumps([chunk_size_bytes, EMBED_MODEL])
hash_value = hashlib.sha256(hash_input).hexdigest()

save_file = f"{hash_value[:7]}-{SENTIMENT_FILE}"

# Function to save progress using pickle
def save_progress():
    with tempfile.NamedTemporaryFile('wb', delete=False) as temp_file:
        pickle.dump({"hash": hash_value, "sentiment_store": sentiment_store}, temp_file)
        temp_file_path = temp_file.name
    os.replace(temp_file_path, save_file)

# Load sentiment data from file if it exists and matches the hash
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
    print("No existing sentiment file found. Starting fresh.")

# Function to extract sentiment score
def extract_sentiment(chunk):
    prompt = f"""Please analyze the following text and provide a rating of the happiness of the author on a scale of 1 to 100. Just provide the numerical rating.

Text:
{chunk}

"""
    response, stats = llm(prompt)
    # Parse the numerical response
    try:
        sentiment_score = int(response.strip())
    except ValueError:
        print('error', response)
        sentiment_score = None
    print(f"Sentiment Score: {sentiment_score}")
    return sentiment_score

# Function to parse date strings
def parse_date(date_str):
    try:
        # Use dateutil.parser to parse the date string
        return parser.parse(date_str)
    except (ValueError, parser.ParserError):
        return None

# Process chunks and extract sentiment scores
chunks_processed = 0
for info in loaded_files:
    date_str = info["date"]
    print(f"Original date string: {date_str}")
    parsed_date = parse_date(date_str)
    if parsed_date is None:
        print(f"Could not parse date: {date_str}")
        continue  # Skip this entry
    content = info["content"]
    corpus_size += len(content)

    chunks = chunkenize_smalloverlap(content, 8192)

    for i, chunk in enumerate(chunks):
        id = f"{date_str}#{i}"
        print(f"Processing chunk ID: {id}")

        chunk_store[id] = date_str + "\n" + chunk

        # Skip if already processed
        if id in sentiment_store:
            if chunks_processed % 5 == 0:
                chunks_processed += 1
            continue

        chunks_processed += 1
        sentiment_score = extract_sentiment(chunk)

        sentiment_store[id] = {
            'date_str': date_str,  # Store date string
            'date': parsed_date,   # Store parsed date
            #'chunk': chunk,
            'sentiment_score': sentiment_score
        }

        # Save progress every 50 chunks
        if chunks_processed % 50 == 0:
            save_progress()

# Save the final progress
save_progress()

preprocessing_timer.stop_and_log(corpus_size)

# Prepare data for visualization
data = []
if not show_year:
    for id, entry in sentiment_store.items():
        if entry['date'] is None:
            print(f"Invalid date for ID {id}, skipping.")
            continue
        data.append({
            'id': id,
            'date': entry['date'],
            'sentiment_score': entry['sentiment_score'],
        })
    
    df = pd.DataFrame(data)
    
    # Assign y_positions within each date group
    def assign_y_positions(group):
        group = group.copy()
        group['y_position'] = range(len(group))
        return group
    
    df = df.groupby('date').apply(assign_y_positions).reset_index(drop=True)
    
    # Convert dates to matplotlib date numbers
    df['date_num'] = mdates.date2num(df['date'])
    
    # Set up the matplotlib figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Use a colormap that goes from red (low sentiment) to green (high sentiment)
    cmap = cm.get_cmap('RdYlGn')
    
    # Plot rectangles for each chunk
    for idx, row in df.iterrows():
        x = row['date_num']
        y = row['y_position']
        #x = 0
        sentiment_score = row['sentiment_score']
        print(x,y,sentiment_score)
        if sentiment_score is not None:
            color = cmap(sentiment_score / 100.0)  # Normalize to 0-1
        else:
            color = 'gray'  # Use gray color if sentiment_score is None
        # Plot a rectangle
        rect = plt.Rectangle((x - 0.4, y), width=0.8, height=0.8, color=color)
        ax.add_patch(rect)
    
    # Configure the x-axis with date labels
    ax.set_xlim(df['date_num'].min() - 1, df['date_num'].max() + 1)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    
    # Set y-axis limits and labels
    ax.set_ylim(-0.5, df['y_position'].max() + 1)
    ax.set_xlabel('Date')
    ax.set_ylabel('Chunks per Entry')
    
    # Add a colorbar to show the sentiment scale
    norm = mcolors.Normalize(vmin=0, vmax=100)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Sentiment Score')
    
    # Display the plot
    plt.title('Sentiment Analysis Over Time')

else:
    for id, entry in sentiment_store.items():
        parsed_date = entry['date']
        if parsed_date is None:
            print(f"Invalid date for ID {id}, skipping.")
            continue
        year = parsed_date.year
        day_of_year = parsed_date.timetuple().tm_yday  # Day of the year (1-366)
        data.append({
                'id': id,
            'year': year,
            'day_of_year': day_of_year,
            'sentiment_score': entry['sentiment_score'],
        })
    
    df = pd.DataFrame(data)
    
    # Assign y_positions within each year group
    def assign_y_positions(group):
        group = group.copy()
        group['y_offset'] = range(len(group))
        return group
    
    df = df.groupby(['year', 'day_of_year']).apply(assign_y_positions).reset_index(drop=True)
    
    # Calculate the overall y_position by combining the year and y_offset
    # We'll use the year number as the base, and add a small fraction for the offset
    df['y_position'] = df['year'] + df['y_offset'] * 0.1  # Adjust the multiplier as needed
    
    # Set up the matplotlib figure and axes
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Use a colormap that goes from red (low sentiment) to green (high sentiment)
    cmap = cm.get_cmap('RdYlGn')
    
    # Plot rectangles for each chunk
    for idx, row in df.iterrows():
        x = row['day_of_year']
        y = row['y_position']
        sentiment_score = row['sentiment_score']
    
        if sentiment_score is not None:
            color = cmap(sentiment_score / 100.0)  # Normalize to 0-1
        else:
            color = 'gray'  # Use gray color if sentiment_score is None
    
        # Plot a rectangle
        rect = plt.Rectangle((x - 0.4, y - 0.05), width=0.8, height=0.1, color=color)
        ax.add_patch(rect)
    
    # Set x-axis limits between 1 and 366 (maximum possible day in a year)
    ax.set_xlim(1, 366)
    
    # Set y-axis labels and limits
    years = sorted(df['year'].unique())
    ax.set_yticks(years)
    ax.set_yticklabels([str(year) for year in years])
    ax.set_ylim(min(years) - 0.5, max(years) + 0.5)
    
    # Set labels
    ax.set_xlabel('Day of the Year')
    ax.set_ylabel('Year')
    
    # Optionally, format x-axis to show months
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    
    # Add a colorbar to show the sentiment scale
    norm = mcolors.Normalize(vmin=0, vmax=100)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Sentiment Score')
    
    # Adjust plot aesthetics
    plt.title('Sentiment Analysis Over Years')
    plt.tight_layout()

plt.show()
