import os
from os import walk
import re
import datetime as dt
import matplotlib.pyplot as plt

from common import loadfiles


def parse_date(date_str):
    """
    Expects date_str in the form YYYY-MM-DD (e.g. '2024-12-24')
    and returns a datetime.datetime object.
    """
    year, month, day = date_str.split('-')
    return dt.datetime(int(year), int(month), int(day))

def main():
    # Load the files from your custom function
    entries = loadfiles()
    # Sort them by date
    entries.sort(key=lambda x: x['date'])

    # ----------------------------
    #  Build x_datetimes and y_sizes
    #  x_datetimes is the parsed date
    #  y_sizes is the length of the file content
    # ----------------------------
    x_datetimes = []
    y_sizes = []
    for item in entries:
        dtime = parse_date(item['date'])
        size = len(item['content'])  # or some other size metric
        x_datetimes.append(dtime)
        y_sizes.append(size)

    # ----------------------------
    #  FIRST PLOT: highlight points in red if
    #  they are the largest within a ±5-file window
    # ----------------------------
    n = len(y_sizes)
    colors = []
    for i in range(n):
        left_idx = max(0, i - 5)
        right_idx = min(n, i + 6)  # slice end is exclusive, so i+6
        window = y_sizes[left_idx:right_idx]

        if y_sizes[i] == max(window):
            colors.append('red')
        else:
            colors.append('blue')

    plt.figure()
    # Optionally draw a thin gray line behind the points
    plt.plot_date(x_datetimes, y_sizes, fmt='-')#, color='lightgray')
    # Scatter the points with the color rule
    plt.scatter(x_datetimes, y_sizes, c=colors)
    plt.title("Daily File Sizes (Red = max within ±5)")

    # Show all plots
    plt.show()

if __name__ == "__main__":
    main()