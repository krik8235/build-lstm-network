# Overview

Predict stock price of MAG-7 by building the LSTM neutral network on Keras framework.

Compare the outcomes when epoch is 10, 100, and 1,000 and plot the results.

1. Extract historical price on Yahoo Finance using Selenium:

![Yahoo Finance](https://res.cloudinary.com/dfeirxlea/image/upload/v1733062828/portfolio/jcexczk8kzromghlhcss.png)


2. Compile and train the LSTM network:

![Terminal image](https://res.cloudinary.com/dfeirxlea/image/upload/v1733069463/portfolio/oetdaic97nn6pou9yvxl.png)


3. The future stock price predicted (Plotted 3 patterns by epoch in red for each stock):

![Prediction](https://res.cloudinary.com/dfeirxlea/image/upload/v1733063005/portfolio/axw0aexisdafvu91c1m3.png)


## Table of Contents

- [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Key Features](#key-features)
  - [Technologies Used](#technologies-used)
  - [Project Structure](#project-structure)
  - [Setup](#setup)
  - [Usage](#usage)
  - [Development](#development)
  - [Contributing](#contributing)
  - [Troubleshooting](#troubleshooting)


## Key Features
Predict future stock price using LSTM networks.

1. **Data Preparation**:
   - Scrape historical stock price from Yahoo Finance using Selenium Webdriver
   - Remove null valeues and store the dataset in the CSV file
   - Run EDA to understand the data structure

2. **Train/Test Data**
   - Take the adj. close price as data frame and split them into train and test dataset
   - Nominalize the datasets from 0 to 1 using the MinMaxScaler preprocessing class from the scikit-learn
   - Split the dataset into train and test data and reform those into NumPy array


3. **Compile LSTM Network**:
   - Compile LSTM network using Keras Sequential framework with 5 dense layers
   - Train the network using datasets

4. **Stock Price Prediction**:
   - Visualize the results on graph. (Compare results by epoch = 10, 100, and 1,000)

5. **Evaluation**:
   - Evaluate the results using RMSE and MAPE.

## Technologies Used

   - Python: Primary programming language. We use ver 3.12

[data-scraping]
   - [Selenium WebDriver](https://www.selenium.dev/documentation/webdriver/): Scrape data from the website
   - [BeatifulSoup4](https://pypi.org/project/beautifulsoup4/): A library that makes it easy to scrape information from web pages
   - [XMLX](https://pypi.org/project/xmlx/): A simple and compact XML parser

[eda]
   - [Matplotlib](https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html): A library for data visualization, typically in the form of plots, graphs and charts.

[ml-stack]
   - [TensorFlow](https://www.tensorflow.org/api_docs/python/tf): A ML/AI software library
   - [keras](https://keras.io/): An open-source API for artifitial neutral network
   - [NumPy](https://numpy.org/): A Python library to operate large, multi-dimensional arrays and matrices
   - [pandas](https://pandas.pydata.org/docs/user_guide/index.html): An open source library with data structures and data analysis tools
   - [scikit-learn](https://pypi.org/project/scikit-learn/): A Python ML module built on top of SciPy 
   - [scikeras](https://adriangb.com/scikeras/stable/): keras x scikit-learn

[deployment]
   - pip, pipenv: Python package manager


## Project Structure
```
.
├── __init__.py
├── predict.py
├── utils/
│   ├── web_scraper.py
│   └── ext_analysis.py
└── sample_data/            # Store the scraped dataset 
└── requirements.txt
```

## Setup

1. Install the `pipenv` package manager:
   ```
   pip install pipenv
   ```

2. Install dependencies:
   ```
   pipenv shell
   pip install -r requirements.txt -v
   ```

## Usage
1. Scrape the latest stock price data:
   ```
   pipenv shell
   python utils/web_scraper.py
   ```
   You will be asked to enter a specific ticker or use default tickers of MAG7.

    <img src="https://res.cloudinary.com/dfeirxlea/image/upload/t_width 300 x height auto/v1733067103/portfolio/lvylanb34gjqs4gq3uka.png">


2. Run EDA:
   ```
   pipenv shell
   python utils/ext_analysis.py
   ```
   You will be asked to select a ticker and title. You can skip them simply pressing enter.

    <img src="https://res.cloudinary.com/dfeirxlea/image/upload/t_width 300 x height auto/v1733067152/portfolio/y3yr24rtmvrrg7i0xyvf.png">


3. Predict stock price:
   ```
   pipenv shell
   python main.py
   ```
   You will be asked to input a ticker or you can skip this by pressing enter. (Default ticker is GOOG)
   

## Development

### Package Management with pipenv

- Add a package: `pipenv install <package>`
- Remove a package: `pipenv uninstall <package>`
- Run a command in the virtual environment: `pipenv run <command>`

* After adding/removing the package, update `requirements.txt` accordingly or run `pip freeze > requirements.txt` to reflect the changes in dependencies.

* To reinstall all the dependencies, delete `Pipfile` and `Pipfile.lock` files, then run:
   ```
   pipenv shell
   pipenv install -r requirements.txt -v
   ```

### Customizing LSTM Network

To customize LSTM, edit the `main.py` file.

### Adding EDA

To add more EDA, edit the `ext_analysis.py` file.


## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-amazing-feature`)
3. Commit your changes (`git commit -m 'Add your-amazing-feature'`)
4. Push to the branch (`git push origin feature/your-amazing-feature`)
5. Open a pull request


## Troubleshooting

Common issues and solutions:
- Memory errors: If processing large contracts, you may need to increase the available memory for the Python process.
- Data scraping issues: Selenium relies on the hard-coded HTML structures. Update `web_scraper.py` accordingly to see if the data was properly scraped.

