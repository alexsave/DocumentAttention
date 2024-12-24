#!/usr/bin/env python3
"""
plot_sentiment_points.py

Loads the sentiment pickle file (my_sentiment.pkl or hashed variant)
and creates a scatter plot (x = date, y = sentiment).
- Points in black for "normal" sentiment
- Points in red for outliers (defined by IQR rule below)
"""

import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# -----------------------------------------------------------------------
# If you used a hashed file name (e.g., "abc1234-my_sentiment.pkl"), 
# set SENTIMENT_FILE to that. Otherwise, it should be "my_sentiment.pkl" 
# by default.
# -----------------------------------------------------------------------
SENTIMENT_FILE = "0e61aa5-my_sentiment.pkl"

def load_sentiment_data(pickle_path):
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Could not find sentiment file: {pickle_path}")
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    # "data" should be a dict with keys: {"hash", "sentiment_store", "summary_store"}
    sentiment_store = data["sentiment_store"]
    return sentiment_store

def main():
    # 1. Load the sentiment data
    sentiment_store = load_sentiment_data(SENTIMENT_FILE)
    
    # 2. Convert the sentiment_store dictionary into a DataFrame
    #    Each entry has:
    #     - 'date' (a datetime object)
    #     - 'sentiment_score' (1–100 or None)
    #     - 'date_str' (unused here)
    rows = []
    for chunk_id, info in sentiment_store.items():
        date_val = info.get("date", None)
        score = info.get("sentiment_score", None)
        if date_val is None or score is None:
            # If date or sentiment missing, skip
            continue
        rows.append({
            "id": chunk_id,
            "date": date_val,
            "sentiment_score": score
        })
    
    df = pd.DataFrame(rows)
    if df.empty:
        print("No valid data to plot. Exiting.")
        return

    # 3. Convert dates to a numeric form recognized by Matplotlib
    df["date_num"] = mdates.date2num(df["date"])

    # 4. Detect outliers by the IQR method:
    #    Outlier = below Q1 - 1.5*IQR or above Q3 + 1.5*IQR
    Q1, Q3 = df["sentiment_score"].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    def is_outlier(x):
        return (x < lower_bound) or (x > upper_bound)

    # Assign colors
    df["color"] = df["sentiment_score"].apply(lambda x: "red" if is_outlier(x) else "black")

    # 5. Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    # Scatter: x = date_num, y = sentiment
    ax.scatter(
        df["date_num"],
        df["sentiment_score"],
        c=df["color"],
        marker="o"
    )

    # 6. Format x-axis as dates
    ax.set_xlim(df["date_num"].min() - 1, df["date_num"].max() + 1)
    ax.xaxis_date()  
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    # 7. Label and show
    ax.set_xlabel("Date")
    ax.set_ylabel("Sentiment Score (1–100)")
    ax.set_title("Sentiment Over Time (Outliers in Red)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
