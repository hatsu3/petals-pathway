import csv
import statistics

# Define a dictionary to store latency data for each client
client_latency = {}

# Read the file and parse each line
with open('e2e_latency.txt', 'r') as csvfile:
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

# Calculate the average and standard deviation for each client
for client_id, latencies in client_latency.items():
    avg_latency = statistics.mean(latencies)
    stdev_latency = -0.0
    if len(latencies) > 1:
        stdev_latency = statistics.stdev(latencies)

    # Print the results
    print(f"Client {client_id}: Average latency = {avg_latency:.2f} ms, Standard deviation = {stdev_latency:.2f} ms")
