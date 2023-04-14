import os
import pandas as pd
import matplotlib.pyplot as plt

###
### Load based routing + Baseline stage assignment
###

load_based_baseline_dir = "./loadbased_baseline/"

# Initialize an empty data frame for the first case
load_based_baseline_csv_files = []

# loop through each CSV file in the directory
for filename in os.listdir(load_based_baseline_dir):
    if filename.endswith(".csv"):
        # read the CSV file into a dataframe
        filepath = os.path.join(load_based_baseline_dir, filename)
        data = pd.read_csv(filepath)
        data.columns = ['client', 'latency']
        load_based_baseline_csv_files.append(data)

load_based_baseline_data = pd.concat(load_based_baseline_csv_files, ignore_index=True, axis=0)
load_based_baseline_total_latency = load_based_baseline_data['latency'].sum()
print(f"Total latency for load based + baseline is: {load_based_baseline_total_latency}")
load_based_baseline_num_requests = len(load_based_baseline_data.index)
print(f"Total number of requests for load based + baseline is: {load_based_baseline_num_requests}")
load_based_baseline_avg_latency = load_based_baseline_total_latency / load_based_baseline_num_requests
print(f"Average latency for load based + baseline is: {load_based_baseline_avg_latency}")

###
### Load based routing + Ideal stage assignment
###

load_based_ideal_dir = "./loadbased_ideal"

# Initialize an empty data frame for the first case
load_based_ideal_csv_files = []

# loop through each CSV file in the directory
for filename in os.listdir(load_based_ideal_dir):
    if filename.endswith(".csv"):
        # read the CSV file into a dataframe
        filepath = os.path.join(load_based_ideal_dir, filename)
        data = pd.read_csv(filepath)
        data.columns = ['client', 'latency']
        load_based_ideal_csv_files.append(data)

load_based_ideal_data = pd.concat(load_based_ideal_csv_files, ignore_index=True, axis=0)
load_based_ideal_total_latency = load_based_ideal_data['latency'].sum()
print(f"Total latency for load based + ideal is: {load_based_ideal_total_latency}")
load_based_ideal_num_requests = len(load_based_ideal_data.index)
print(f"Total number of requests for load based + ideal is: {load_based_ideal_num_requests}")
load_based_ideal_avg_latency = load_based_ideal_total_latency / load_based_ideal_num_requests
print(f"Average latency for load based + ideal is: {load_based_ideal_avg_latency}")

###
### Load based routing + Request rate stage assignment
###

load_based_request_rate_dir = "./loadbased_requestrate"

# Initialize an empty data frame for the first case
load_based_request_rate_csv_files = []

# loop through each CSV file in the directory
for filename in os.listdir(load_based_request_rate_dir):
    if filename.endswith(".csv"):
        # read the CSV file into a dataframe
        filepath = os.path.join(load_based_request_rate_dir, filename)
        data = pd.read_csv(filepath)
        data.columns = ['client', 'latency']
        load_based_request_rate_csv_files.append(data)

load_based_request_rate_data = pd.concat(load_based_request_rate_csv_files, ignore_index=True, axis=0)
load_based_request_rate_total_latency = load_based_request_rate_data['latency'].sum()
print(f"Total latency for load based + request rate is: {load_based_request_rate_total_latency}")
load_based_request_rate_num_requests = len(load_based_request_rate_data.index)
print(f"Total number of requests for load based + request rate is: {load_based_request_rate_num_requests}")
load_based_request_rate_avg_latency = load_based_request_rate_total_latency / load_based_request_rate_num_requests
print(f"Average latency for load based + request rate is: {load_based_request_rate_avg_latency}")

###
### Random routing + Baseline stage assignment
###

random_baseline_dir = "./random_baseline"

# Initialize an empty data frame for the first case
random_baseline_csv_files = []

# loop through each CSV file in the directory
for filename in os.listdir(random_baseline_dir):
    if filename.endswith(".csv"):
        # read the CSV file into a dataframe
        filepath = os.path.join(random_baseline_dir, filename)
        data = pd.read_csv(filepath)
        data.columns = ['client', 'latency']
        random_baseline_csv_files.append(data)

random_baseline_data = pd.concat(random_baseline_csv_files, ignore_index=True, axis=0)
random_baseline_total_latency = random_baseline_data['latency'].sum()
print(f"Total latency for random + baseline is: {random_baseline_total_latency}")
random_baseline_num_requests = len(random_baseline_data.index)
print(f"Total number of requests for random + baseline is: {random_baseline_num_requests}")
random_baseline_avg_latency = random_baseline_total_latency / random_baseline_num_requests
print(f"Average latency for random + baseline is: {random_baseline_avg_latency}")

###
### Random routing + Ideal stage assignment
###

random_ideal_dir = "./random_ideal"

# Initialize an empty data frame for the first case
random_ideal_csv_files = []

# loop through each CSV file in the directory
for filename in os.listdir(random_ideal_dir):
    if filename.endswith(".csv"):
        # read the CSV file into a dataframe
        filepath = os.path.join(random_ideal_dir, filename)
        data = pd.read_csv(filepath)
        data.columns = ['client', 'latency']
        random_ideal_csv_files.append(data)

random_ideal_data = pd.concat(random_ideal_csv_files, ignore_index=True, axis=0)
random_ideal_total_latency = random_ideal_data['latency'].sum()
print(f"Total latency for random + ideal is: {random_ideal_total_latency}")
random_ideal_num_requests = len(random_ideal_data.index)
print(f"Total number of requests for random + ideal is: {random_ideal_num_requests}")
random_ideal_avg_latency = random_ideal_total_latency / random_ideal_num_requests
print(f"Average latency for random + ideal is: {random_ideal_avg_latency}")

###
### Random routing + Request rate stage assignment
###

random_request_rate_dir = "./random_requestrate"

# Initialize an empty data frame for the first case
random_request_rate_csv_files = []

# loop through each CSV file in the directory
for filename in os.listdir(random_request_rate_dir):
    if filename.endswith(".csv"):
        # read the CSV file into a dataframe
        filepath = os.path.join(random_request_rate_dir, filename)
        data = pd.read_csv(filepath)
        data.columns = ['client', 'latency']
        random_request_rate_csv_files.append(data)

random_request_rate_data = pd.concat(random_request_rate_csv_files, ignore_index=True, axis=0)
random_request_rate_total_latency = random_request_rate_data['latency'].sum()
print(f"Total latency for random + request rate is: {random_request_rate_total_latency}")
random_request_rate_num_requests = len(random_request_rate_data.index)
print(f"Total number of requests for random + request rate is: {random_request_rate_num_requests}")
random_request_rate_avg_latency = random_request_rate_total_latency / random_request_rate_num_requests
print(f"Average latency for random + request rate is: {random_request_rate_avg_latency}")

###
### Plotting
###

# Figure Size
fig = plt.figure(figsize =(10, 10))

# Create the list of policies and the latencies
policies = [
    "LB-B",
    "LB-I",
    "LB-RR",
    "R-B",
    "R-I",
    "R-RR"
]

latencies = [
    load_based_baseline_avg_latency,
    load_based_ideal_avg_latency,
    load_based_request_rate_avg_latency,
    random_baseline_avg_latency,
    random_ideal_avg_latency,
    random_request_rate_avg_latency
]

colors = [
    "blue",
    "blue",
    "blue",
    "red",
    "red",
    "red"
]

plt.bar(policies, latencies, color=colors)

# Add labels and a title. Note the use of `labelpad` and `pad` to add some
# extra space between the text and the tick labels.
plt.xlabel('Policy')
plt.ylabel('Average Latency (sec)')
plt.title('Policy Effects on Latency')

plt.savefig('./policy-vs-latency.png')
