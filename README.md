# Forecast Electricity Load Peaks

This repository contains the code to forecast the probability of Electricity Load Peaks. The project involves data loading, processing, and analysis of electricity load data, along with weather data and holiday information.

## Setup

To set up the project, follow these steps:

1. **Clone the repository:**

    ```sh
    git clone <repository-url>
    cd forecast_peak
    ```

2. **Install dependencies:**

    This project uses [Poetry](https://python-poetry.org/) for dependency management. Ensure you have Poetry installed, then run:

    ```sh
    poetry install
    ```

3. **Activate the virtual environment:**

    ```sh
    poetry shell
    ```

## Usage

### Running the Jupyter Notebook

To run the Jupyter notebook for data analysis:

1. **Start Jupyter Notebook:**

    ```sh
    jupyter notebook
    ```

2. **Open `assignement.ipynb` in the Jupyter interface.**

### Running the Probability Estimation Script

To run the probability estimation script:

```sh
python estimate_probability.py
```

