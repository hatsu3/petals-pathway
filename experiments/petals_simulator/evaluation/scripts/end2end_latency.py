import os
import sys
import csv
import statistics

import numpy as np


# Define a dictionary to store latency data for each client
client_latency = {}

if len(sys.argv) < 2:
    print('Please provide the path to the directory containing the CSV files as a command line argument')
    sys.exit()

# extract the directory path from the command line argument
directory = os.path.abspath(sys.argv[1])

# Loop through all the files in the folder
for filename in os.listdir(directory):
    # Read the file and parse each line
    with open(os.path.join(directory, filename), 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            client_id = int(row[0])
            latency = float(row[1])

            # Check if the client is already in the dictionary
            if client_id in client_latency:
                # If the client is in the dictionary, add the latency to the list of latencies
                client_latency[client_id].append(latency)
            else:
                # If the client is not in the dictionary, create a new list with the latency as the first element
                client_latency[client_id] = [latency]

p99s = []

# Calculate the average and standard deviation for each client
for client_id, latencies in client_latency.items():
    avg_latency = statistics.mean(latencies)
    stdev_latency = -0.0
    if len(latencies) > 1:
        stdev_latency = statistics.stdev(latencies)

    p99_latency = np.percentile(latencies, 99)
    p99s.append(p99_latency)

    # # Print the results
    # print(f"Client {client_id}: Average latency = {avg_latency:.2f} s, "
    #       f"Standard deviation = {stdev_latency:.2f} s, 99th percentile = {p99_latency:.2f} s")

avg_clients = statistics.mean(p99s)
stdev_clients = statistics.stdev(p99s)
print(f"avg: {avg_clients:.2f}  stddev: {stdev_clients:.2f} s")
