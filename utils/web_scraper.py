import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By


def scrape_historical_price(ticker):
    if not ticker:
        return 

    driver = webdriver.Chrome()
    driver.maximize_window()

    driver.get(f"https://finance.yahoo.com/quote/{ticker}/history/")
    
    # i = 1
    # while i < 100:
    #     driver.execute_script(
    #         f"window.scrollTo(0, (document.body.scrollHeight)*{i * 10});"
    #     )
    #     i = i + 1
    #     time.sleep(0.5)

    data = []
    webpage = driver.page_source
    soup = BeautifulSoup(webpage, "lxml")
    stock_prices = soup.find_all("tr", class_="yf-j5d1ld")

    for stock_price in stock_prices:
        stock_price = stock_price.find_all("td")
        price = []
        for stock in stock_price:
            price.append(stock.text)
        data.append(price)

    database = pd.DataFrame(
        data,
        columns=["date", "open", "high", "low", "close", "adj_close", "volume"],
    )
    database.to_csv(rf"sample_data/{ticker}.csv", index=False)
    driver.quit()



def prepare_data(csv_file_name):
    data = pd.read_csv(f"./sample_data/{csv_file_name}.csv")
    data.head()
    data.info()

    # drop null cells
    data.dropna(inplace=True)

    # set up dtype
    data["date"] = pd.to_datetime(data["date"])

    for column in ["open", "high", "low", "close", "adj_close", "volume"]:
        # if isinstance(data[column], str):
        data[column] = data[column].astype(str).str.replace(",", "", regex=True)

    # data = data.astype(
    #     {
    #         "open": "float32",
    #         "high": "float32",
    #         "low": "float32",
    #         "close": "float32",
    #         "adj_close": "float32",
    #         "volume": "float32",
    #     }
    # )

    # sort by date
    data = data.sort_values(by="date", ignore_index=True, ascending=False)

    # # drop old data
    # data = data[data["date"] >= "2015-01-01"].reset_index(drop=True)

    # add moving averages
    moving_ave_targets = [10, 20, 50]
    for target in moving_ave_targets:
        column_title = f"{target}_day_moving_ave."
        data[column_title] = data["adj_close"].rolling(target).mean()

    data.describe()
    return data


if __name__ == "__main__":
    print("Start to scrape historical stock price.")
    print("Input a ticker. (Or press enter to employ the default - MAG7)")
    ticker = input()
    default_ticker_list = ["GOOG", "AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META"]

    if not ticker:
        for item in default_ticker_list:
            print(f"Start to scrape the data of {item}")
            scrape_historical_price(ticker=item)
    
    else:
        scrape_historical_price(ticker=ticker)

    print("Completed.")

    