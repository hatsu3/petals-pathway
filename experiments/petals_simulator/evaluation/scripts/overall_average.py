import os
import sys
import csv
import statistics

import numpy as np


# Define a dictionary to store latency data for each client
latency = []

if len(sys.argv) < 2:
    print('Please provide the path to the directory containing the CSV files as a command line argument')
    sys.exit()

# extract the directory path from the command line argument
directory = os.path.abspath(sys.argv[1])

parts = sys.argv[1].split("/")[-1].split("_")
num_clients = int(parts[-1])

# Loop through all the files in the folder
for filename in os.listdir(directory):
    # Read the file and parse each line
    with open(os.path.join(directory, filename), 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            latency.append(float(row[1]))

print(f"{num_clients}, {statistics.mean(latency)}")
