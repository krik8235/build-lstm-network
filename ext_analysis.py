import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from web_scraper import prepare_data

warnings.filterwarnings("ignore")

NVDA_data = prepare_data("NVDA")
datasets = [
    {"data": NVDA_data, "title": "NVIDIA"},
]


def plot_graph(data_point):
    plt.figure(figsize=(20, 12))
    for i in range(0, len(datasets)):
        data = datasets[i]["data"]
        plt.subplot(3, 2, i + 1)
        plt.plot(data["date"], data[data_point])
        plt.title(datasets[i]["title"])
        plt.ylabel(data_point)
        plt.show()


# plot Adj. Close
plot_graph("adj_close")
