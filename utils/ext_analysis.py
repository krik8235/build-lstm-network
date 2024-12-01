import warnings
import matplotlib.pyplot as plt
from web_scraper import prepare_data

warnings.filterwarnings("ignore")


def make_dt(ticker="NVDA", title="NVIDIA"):
    data = prepare_data(ticker)
    dt = [{ "data": data, "title": title }]
    return dt


def plot_graph(data_point, ticker="NVDA", title="NVIDIA"):
    plt.figure(figsize=(20, 12))
    dt = make_dt(ticker, title)
    for i in range(0, len(dt)):
        data = dt[i]["data"]
        plt.subplot(3, 2, i + 1)
        plt.plot(data["date"], data[data_point])
        plt.title(dt[i]["title"])
        plt.ylabel(data_point)
        plt.show()


if __name__ == "__main__":
    print("Input ticker (Press enter to skip)")
    ticker = input() if input() else "NVDA"

    print("Enter the title of the graph. (Press enter to skip)")
    title = input() if input() else "NVIDIA"

    plot_graph("adj_close", ticker=ticker, title=title)
