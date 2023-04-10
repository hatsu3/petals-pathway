import os
import random
import pickle as pkl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from geopy import Point
from geopy.distance import great_circle

# TODO: load network latency dataset (e.g. RIPE Atlas, CAIDA)

def load_verizon_dataset():
    # central locations of regions
    region_locs = (pd.read_csv("data/region_locs.csv")
                   .set_index("Region")[["Latitude", "Longitude"]]
                   .apply(tuple, axis=1).to_dict())

    verizon_df = pd.read_csv("data/verizon_lat_stat.csv")
    
    # calculate distance between src and dst
    verizon_df["distance"] = (verizon_df.apply(
        lambda row: great_circle(
            region_locs[row["src"]], 
            region_locs[row["dst"]]
        ).miles, axis=1))
    
    # drop src and dst columns
    verizon_df.drop(columns=["src", "dst"], inplace=True)

    # calculate the average latency across all months
    verizon_df["avg_latency"] = verizon_df.iloc[:, :-1].mean(axis=1)

    # drop all the month columns
    verizon_df.drop(columns=verizon_df.columns[:-2], inplace=True)

    return verizon_df


def fit_linreg(test_size=0.2, random_state=42):
    verizon_df = load_verizon_dataset()
    
    # split the dataset into training and testing sets
    train_df = verizon_df.sample(frac=1-test_size, random_state=random_state)
    test_df = verizon_df.drop(train_df.index)

    # fit a linear regression model
    linreg = LinearRegression()
    linreg.fit(train_df[["distance"]], train_df["avg_latency"])

    # plot the model
    plt.scatter(train_df["distance"], train_df["avg_latency"], color="black")
    plt.scatter(test_df["distance"], test_df["avg_latency"], color="red")
    plt.plot(train_df["distance"], linreg.predict(train_df[["distance"]]), color="blue", linewidth=3)
    plt.xlabel("Distance (miles)")
    plt.ylabel("Latency (ms)")
    
    # save the plot
    plt.savefig("data/latency_vs_distance.png")
    
    # evaluate the model
    print(f"R^2: {linreg.score(test_df[['distance']], test_df['avg_latency'])}")

    return linreg


def generate_random_location(within_us=True):
    if within_us:
        lat = random.uniform(24.396308, 49.384358)  # Latitude range for the US
        lon = random.uniform(-124.848974, -66.885444)  # Longitude range for the US
    else:
        lat = random.uniform(-90, 90)
        lon = random.uniform(-180, 180)
    return Point(lat, lon)


class LatencyEstimator:
    def __init__(self, estimator=None):
        self._estimator = estimator

    def fit(self, distances, latencies):
        if self._estimator is None:
            self._estimator = LinearRegression()
        distances = np.array(distances).reshape(-1, 1)
        latencies = np.array(latencies).reshape(-1, 1)
        assert distances.shape == latencies.shape
        self._estimator.fit(distances, latencies)

    def predict(self, src: Point, dst: Point):
        if self._estimator is None:
            raise ValueError("Please fit the estimator first.")
        distance = great_circle(src, dst).miles
        return self._estimator.predict(np.array([[distance]]))[0]
    
    def score(self, distances, latencies):
        if self._estimator is None:
            raise ValueError("Please fit the estimator first.")
        distances = np.array(distances).reshape(-1, 1)
        latencies = np.array(latencies).reshape(-1, 1)
        assert distances.shape == latencies.shape
        return self._estimator.score(distances, latencies)
    
    def save(self, filename="data/latency_estimator.pkl"):
        with open(filename, "wb") as f:
            pkl.dump(self._estimator, f)
    
    @classmethod
    def load(cls, filename="data/latency_estimator.pkl"):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"{filename} does not exist.")
        with open(filename, "rb") as f:
            return cls(pkl.load(f))


if __name__ == "__main__":
    load_verizon_dataset()
    fit_linreg()
