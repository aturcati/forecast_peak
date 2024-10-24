from pathlib import Path

import pandas as pd
import pickle
import numpy as np
import warnings


import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV

from loguru import logger

import holidays

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def add_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a datetime index to the dataframe by combining the 'Date' and 'Hour' columns.

    Args:
        df (pd.DataFrame): The dataframe with 'Date' and 'Hour' columns.

    Returns:
        pd.DataFrame: The dataframe with a new datetime index.
    """
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df["Datetime"] = df["Date"] + pd.to_timedelta(df["Hour"] - 1, unit="h")
    df.set_index("Datetime", inplace=True)
    return df


def expanding_window_cross_validation(
    df: pd.DataFrame,
    initial_train_years: int = 1,
    years_in_data: int = 3,
    first_year: int = 2005,
) -> list:
    """
    Create expanding window cross-validation folds.

    Args:
        df (pd.DataFrame): The dataframe to split into training and testing sets.
        initial_train_years (int, optional): Number of initial years for training. Default is 1.
        years_in_data (int, optional): Total number of years in the dataset. Default is 3.
        first_year (int, optional): The first year in the dataset. Default is 2005.

    Returns:
        list: A list of tuples, each containing the training and testing sets for a fold.
    """
    folds = []
    for year in range(initial_train_years, years_in_data):
        train_end_date = pd.Timestamp(f"{first_year + year}-12-31 23:00:00")
        test_start_date = pd.Timestamp(f"{first_year + year + 1}-01-01 00:00:00")
        test_end_date = pd.Timestamp(f"{first_year + year + 1}-12-31 23:00:00")

        train = df.loc[:train_end_date].copy()
        test = df.loc[test_start_date:test_end_date].copy()

        folds.append((train, test))

    return folds


def feature_engineering_temporal(
    df: pd.DataFrame, holidays: set = None
) -> pd.DataFrame:
    """
    Perform temporal feature engineering on the load data.

    Args:
        df (pd.DataFrame): The dataframe to add temporal features.
        holidays (set, optional): A set of holiday dates to mark holidays in the data. Default is None.

    Returns:
        pd.DataFrame: The dataframe with new temporal features.
    """
    df["Dayofweek"] = df.index.dayofweek.astype("int")
    df["Dayofyear"] = df.index.dayofyear.astype("int")
    df["sin_Dayofyear"] = sin_encoding(df, "Dayofyear")
    df["Week"] = df.index.isocalendar().week.astype("int")
    df["Month"] = df.index.month.astype("int")
    df["Quarter"] = df.index.quarter.astype("int")
    df["Year"] = df.index.year.astype("int")
    if holidays is not None:
        df["Holiday"] = [1 if date in holidays else 0 for date in df.index.date]

    return df


def feature_engineering_template(
    df: pd.DataFrame, template: pd.DataFrame
) -> pd.DataFrame:
    """
    Perform feature engineering using a template for the load data.

    Args:
        df (pd.DataFrame): The dataframe to modify.
        template (pd.DataFrame): The template dataframe to merge features.

    Returns:
        pd.DataFrame: The modified dataframe with new features.
    """
    temp_index = df.index.copy()
    df = df.merge(template, how="left", on=["Month", "Dayofweek", "Hour"])
    df.index = temp_index
    return df


def feature_engineering_lags(
    df: pd.DataFrame, columns: list, lag_start: int, lag_end: int
) -> pd.DataFrame:
    """
    Generate lag features for the specified columns in the dataframe.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        columns (list): The columns for which to generate lag features.
        lag_start (int): The start of the lag window.
        lag_end (int): The end of the lag window.

    Returns:
        pd.DataFrame: The dataframe with lag features added.
    """
    for station in columns:
        for i in range(lag_start, lag_end):
            df[f"{station}_lag_{i}"] = df[station].shift(i)
    return df.bfill()


def feature_engineering_lags_shifts(df: pd.DataFrame, shifts: dict) -> pd.DataFrame:
    """
    Generate lag features with predefined shifts for the specified columns.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        shifts (dict): A dictionary where keys are column names and values are shift amounts.

    Returns:
        pd.DataFrame: The dataframe with lag features added.
    """
    for c, val in shifts.items():
        for i in [0, 1, 2, 3, 10, 11, 12, 13, 23, 24, 25]:
            df[f"{c}_lag_{val+i}"] = df[c].shift(val + i)
    return df.bfill()


def feature_engineering_leads(
    df: pd.DataFrame, columns: list, lead_start: int, lead_end: int
) -> pd.DataFrame:
    """
    Generate lead features for the specified columns in the dataframe.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        columns (list): The columns for which to generate lead features.
        lead_start (int): The start of the lead window.
        lead_end (int): The end of the lead window.

    Returns:
        pd.DataFrame: The dataframe with lead features added.
    """
    for station in columns:
        for i in range(lead_start, lead_end):
            df[f"{station}_lead_{i}"] = df[station].shift(-i)
    return df.ffill()


def feature_engineering_rolling(
    df: pd.DataFrame, columns: list, window: int
) -> pd.DataFrame:
    """
    Generate rolling max features for the specified columns.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        columns (list): The columns for which to generate rolling features.
        window (int): The rolling window size.

    Returns:
        pd.DataFrame: The dataframe with rolling max features added.
    """
    for station in columns:
        df[f"{station}_rolling_max_{window}"] = df[station].rolling(window).max()
    return df.bfill()


def score(prediction: np.ndarray, actual: np.ndarray) -> float:
    """
    Calculate the score of a prediction based on sum of squared differences.

    Args:
        prediction (np.ndarray): The predicted values.
        actual (np.ndarray): The actual values.

    Returns:
        float: The score calculated as sum of squared differences.
    """
    return np.sum((prediction - actual) ** 2)


def sin_encoding(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Perform sinusoidal encoding of a column in the dataframe.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        column (str): The column to encode.

    Returns:
        pd.Series: The sinusoidal encoding of the specified column.
    """
    return np.sin(2 * np.pi * df[column] / df[column].max())


def template_single_column(
    dataframe: pd.DataFrame,
    column: str,
    grouping_columns: str,
    methods: list = ["mean"],
    change_col_names: bool = True,
    datetime_index: bool = False,
    index_format: str = None,
) -> pd.DataFrame:
    """
    Create template features for a single column by applying aggregation methods to groups.

    Args:
        dataframe (pd.DataFrame): The dataframe to modify.
        column (str): The column to apply the aggregation method on.
        grouping_columns (str): Columns used to group the dataframe.
        methods (list, optional): List of aggregation methods to apply (e.g., 'mean'). Default is ["mean"].
        change_col_names (bool, optional): Whether to change column names to include the method name. Default is True.
        datetime_index (bool, optional): Whether to convert the resulting index to datetime. Default is False.
        index_format (str, optional): The format to use when converting index to datetime. Required if datetime_index is True.

    Returns:
        pd.DataFrame: The dataframe with new template features generated by aggregating the specified column.
    """
    df_group = dataframe.groupby(grouping_columns)
    templates_dict = {}
    for _, met in enumerate(methods):
        df_group_m = eval(f"df_group.{met}()[column]")

        name = f"{column}_tmpl_{met}" if change_col_names else column

        templates_dict[name] = df_group_m

    df_group = pd.DataFrame(templates_dict)

    if len(grouping_columns) > 1:
        levels = []
        index_name = ""
        for ic, col in enumerate(grouping_columns):
            levels += [ic]
            if ic == 0:
                index_name += "{0[" + col + "]:.0f}"
            else:
                index_name += "-{0[" + col + "]:.0f}"
        df_group.reset_index(level=levels, inplace=True)
        df_group["group_index"] = df_group.agg(index_name.format, axis=1)
        df_group.set_index("group_index", drop=True, inplace=True)
    else:
        df_group.index.rename("group_index", inplace=True)
        df_group[grouping_columns[0]] = df_group.index

    if datetime_index:
        df_group.index = pd.to_datetime(df_group.index, format=index_format)
        df_group.index.names = ["Timestamp"]

    return df_group


def tune_catboost(X_train: pd.DataFrame, y_train: pd.Series) -> CatBoostClassifier:
    """
    Perform hyperparameter tuning for CatBoost model using GridSearchCV.

    Args:
        X_train (pd.DataFrame): The training feature matrix.
        y_train (pd.Series): The training target vector.

    Returns:
        CatBoostClassifier: The best CatBoost classifier after hyperparameter tuning.
    """
    cat_param_grid = {
        "iterations": [500, 1000],
        "learning_rate": [0.01, 0.05, 0.1],
        "depth": [4, 6, 8],
        "l2_leaf_reg": [3, 5, 7],
        "border_count": [32, 64, 128],
    }

    cat_clf = CatBoostClassifier(verbose=0)

    grid_search_cat = GridSearchCV(
        estimator=cat_clf,
        param_grid=cat_param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
    )

    grid_search_cat.fit(X_train, y_train)

    logger.info(f"Best parameters: {grid_search_cat.best_params_}")

    return grid_search_cat.best_estimator_


def tune_lgbm(X_train: pd.DataFrame, y_train: pd.Series) -> lgb.LGBMClassifier:
    """
    Perform hyperparameter tuning for LightGBM model using GridSearchCV.

    Args:
        X_train (pd.DataFrame): The training feature matrix.
        y_train (pd.Series): The training target vector.

    Returns:
        LGBMClassifier: The best LightGBM classifier after hyperparameter tuning.
    """
    lgb_param_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [20, 31, 40],
        "max_depth": [-1, 5, 7],
        "subsample": [0.7, 0.8, 1.0],
    }

    lgb_clf = lgb.LGBMClassifier(verbose=-1)

    grid_search_lgb = GridSearchCV(
        estimator=lgb_clf,
        param_grid=lgb_param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
    )

    grid_search_lgb.fit(X_train, y_train)

    logger.info(f"Best parameters: {grid_search_lgb.best_params_}")

    return grid_search_lgb.best_estimator_


def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    """
    Perform hyperparameter tuning for XGBoost model using GridSearchCV.

    Args:
        X_train (pd.DataFrame): The training feature matrix.
        y_train (pd.Series): The training target vector.

    Returns:
        XGBClassifier: The best XGBoost classifier after hyperparameter tuning.
    """
    xgb_param_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    }

    xgb_clf = xgb.XGBClassifier()

    grid_search_xgb = GridSearchCV(
        estimator=xgb_clf,
        param_grid=xgb_param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
    )

    grid_search_xgb.fit(X_train, y_train)

    logger.info(f"Best parameters: {grid_search_xgb.best_params_}")

    return grid_search_xgb.best_estimator_


def estimate_probability(
    columns_feateng: dict = None, test: bool = True, tune: bool = False
) -> tuple[list, pd.DataFrame]:
    """
    Estimate the probability of peak load occurrence using weather and load data.

    This function processes weather and load historical data to estimate the probability
    of each hour being a peak load hour. It performs feature engineering, trains machine
    learning models (XGBoost, LightGBM, CatBoost), and optionally tunes the models.
    The function supports expanding window cross-validation to evaluate the model on
    different time periods and can return either the test results or estimated probabilities
    for future periods.

    Args:
        columns_feateng (dict, optional): A dictionary containing lists of column names
            for feature engineering. These lists correspond to different lag periods and
            rolling windows for the stations' weather data. Default is None, which triggers
            the use of a predefined set of stations and lag values.
        test (bool, optional): If True, the function performs testing using cross-validation
            on historical data. If False, it applies the trained model to predict future probabilities.
            Default is True.
        tune (bool, optional): If True, the function tunes the model hyperparameters using GridSearchCV
            before training the models. If False, default model configurations are used. Default is False.

    Returns:
        tuple[list, pd.DataFrame]:
            - A list of training columns used for feature engineering.
            - A DataFrame containing the probability estimates or the test results, depending on the value of `test`.
    """
    data_dir = Path("./data")

    # Define the US holidays
    us_holidays = holidays.US()

    logger.info("Reading the data")
    # Read the load data
    load = pd.read_csv(data_dir / "load_hist_data.csv")
    load = add_index(load)
    # Add peak load flag
    load["is_peak"] = (
        load.groupby("Date")["Load"].transform(lambda x: x == x.max()).astype(int)
    )

    # Read the weather data
    weather = pd.read_csv(data_dir / "weather_data.csv")

    # Create a pivot table with the temperature data
    weather_pivot = (
        add_index(weather)
        .reset_index()
        .pivot_table(index="Datetime", columns="Station ID", values="Temperature")
    )
    weather_pivot.columns = [f"Station {col}" for col in weather_pivot.columns]

    # Merge the weather and load data
    merged_data = pd.merge(
        weather_pivot, load, left_index=True, right_index=True, how="inner"
    )

    # Expanding window cross-validation
    folds = expanding_window_cross_validation(
        merged_data, initial_train_years=0, years_in_data=3, first_year=2005
    )

    # Define the target column
    target = "is_peak"

    # Feature engineering columns:
    if columns_feateng is None:
        columns_feateng = {
            "0": [
                "Station 1",
                "Station 14",
                "Station 15",
                "Station 17",
                "Station 19",
                "Station 22",
                "Station 28",
            ],
            "12": [
                "Station 11",
                "Station 16",
                "Station 20",
                "Station 22",
                "Station 24",
                "Station 6",
            ],
            "24": [
                "Station 1",
                "Station 17",
                "Station 19",
                "Station 2",
                "Station 22",
                "Station 28",
                "Station 5",
            ],
            "roll": [
                "Station 4",
                "Station 11",
                "Station 14",
                "Station 19",
                "Station 17",
            ],
        }

    if test:
        logger.info("Testing the model")
        # Expanding window cross-validation
        for i, (train, test) in enumerate(folds[:-1]):
            # Feature engineering
            logger.info("Feature engineering training data")
            train = feature_engineering_temporal(train, holidays=us_holidays)

            train = feature_engineering_lags(
                train, columns=columns_feateng["0"], lag_start=1, lag_end=6
            )
            train = feature_engineering_lags(
                train, columns=columns_feateng["12"], lag_start=12, lag_end=14
            )
            train = feature_engineering_lags(
                train, columns=columns_feateng["24"], lag_start=24, lag_end=26
            )
            train = feature_engineering_rolling(
                train, columns=columns_feateng["roll"], window=4
            )

            # Drop the columns that are not needed
            train = train.drop(columns=["Date", "Load", "Dayofyear"])

            # Repeat the feature engineering for the test set
            logger.info("Feature engineering testing data")
            test = feature_engineering_temporal(test, holidays=us_holidays)
            test = feature_engineering_lags(
                test, columns=columns_feateng["0"], lag_start=1, lag_end=6
            )
            test = feature_engineering_lags(
                test, columns=columns_feateng["12"], lag_start=12, lag_end=14
            )
            test = feature_engineering_lags(
                test, columns=columns_feateng["24"], lag_start=24, lag_end=26
            )
            test = feature_engineering_rolling(
                test, columns=columns_feateng["roll"], window=4
            )
            test = test.drop(columns=["Date", "Load", "Dayofyear"])

            # Define the training columns
            training_columns = train.columns.drop(target)

            # Split the data into training and testing sets
            X_train, y_train = train[training_columns], train[target]
            X_test, y_test = test[training_columns], test[target]

            # Classifiers
            if tune:
                xgb_clf = tune_xgboost(X_train, y_train)
                lgb_clf = tune_lgbm(X_train, y_train)
                cat_clf = tune_catboost(X_train, y_train)
            else:
                xgb_clf = xgb.XGBClassifier()
                lgb_clf = lgb.LGBMClassifier(verbose=-1)
                cat_clf = CatBoostClassifier(verbose=0)

            # Define voting ensemble
            ensemble_clf = VotingClassifier(
                estimators=[("xgb", xgb_clf), ("lgb", lgb_clf), ("cat", cat_clf)],
                voting="soft",
            )

            # Train the model
            logger.info("Training the model")
            ensemble_clf.fit(X_train, y_train)

            # Get probabilistic predictions (probabilities for each class)
            probabilities = ensemble_clf.predict_proba(X_test)[:, 1]
            probabilities[probabilities < 1e-2] = 0.0

            test["Date"] = test.index.date
            test["class_probability"] = np.round(probabilities, 2)
            test["normalized_probability"] = test.groupby("Date")[
                "class_probability"
            ].transform(lambda x: x / x.sum())
            test["predicted_is_peak"] = (
                test.groupby("Date")["normalized_probability"]
                .transform(lambda x: x == x.max())
                .astype(int)
            )

            logger.info("Saving the benchmark results")
            test.to_csv(data_dir / "results" / f"fold_{i+1}_results.csv")

            logger.info(
                f"Fold {i+1}: Score {score(test["normalized_probability"], y_test)}"
            )
        return training_columns, test
    else:
        train, _ = folds[-1]

        # Feature engineering
        logger.info("Feature engineering full training data")
        train = feature_engineering_temporal(train, holidays=us_holidays)

        train = feature_engineering_lags(
            train, columns=columns_feateng["0"], lag_start=1, lag_end=6
        )
        train = feature_engineering_lags(
            train, columns=columns_feateng["12"], lag_start=12, lag_end=14
        )
        train = feature_engineering_lags(
            train, columns=columns_feateng["24"], lag_start=24, lag_end=26
        )
        train = feature_engineering_rolling(
            train, columns=columns_feateng["roll"], window=4
        )

        # Drop the columns that are not needed
        train = train.drop(columns=["Date", "Load", "Dayofyear"])

        # Define the training columns
        training_columns = train.columns.drop(target)

        # Define the training set
        X_train, y_train = train[training_columns], train[target]

        # Classifiers
        if tune:
            xgb_clf = tune_xgboost(X_train, y_train)
            lgb_clf = tune_lgbm(X_train, y_train)
            cat_clf = tune_catboost(X_train, y_train)
        else:
            xgb_clf = xgb.XGBClassifier()
            lgb_clf = lgb.LGBMClassifier(verbose=-1)
            cat_clf = CatBoostClassifier(verbose=0)

        # Define voting ensemble
        ensemble_clf = VotingClassifier(
            estimators=[("xgb", xgb_clf), ("lgb", lgb_clf), ("cat", cat_clf)],
            voting="soft",
        )

        # Train the model
        logger.info("Training the model")
        ensemble_clf.fit(X_train, y_train)

        # Saving the model to disk
        logger.info("Saving the model")
        with open(data_dir / "models" / "model.pkl", "wb") as f:
            pickle.dump(ensemble_clf, f)

        # Load data to estimate
        logger.info("Reading the data to estimate")
        estimates = pd.read_csv(data_dir / "probability_estimates.csv")
        estimates = add_index(estimates)

        # Feature engineering
        logger.info("Feature engineering forecast data")
        forecast_data = weather_pivot.copy()
        forecast_data["Hour"] = forecast_data.index.hour
        forecast_data = feature_engineering_temporal(
            forecast_data, holidays=us_holidays
        )
        forecast_data = feature_engineering_lags(
            forecast_data, columns=columns_feateng["0"], lag_start=1, lag_end=6
        )
        forecast_data = feature_engineering_lags(
            forecast_data, columns=columns_feateng["12"], lag_start=12, lag_end=14
        )
        forecast_data = feature_engineering_lags(
            forecast_data, columns=columns_feateng["24"], lag_start=24, lag_end=26
        )
        forecast_data = feature_engineering_rolling(
            forecast_data, columns=columns_feateng["roll"], window=4
        )
        forecast_data = forecast_data.loc[estimates.index]
        forecast_data.loc[estimates.index, "Hour"] = estimates.loc[
            estimates.index, "Hour"
        ].astype("int32")
        forecast_data.loc[estimates.index, "Date"] = estimates.loc[
            estimates.index, "Date"
        ]

        # Get probabilistic predictions
        logger.info("Estimating the probabilities")
        probabilities = ensemble_clf.predict_proba(forecast_data[training_columns])[
            :, 1
        ]
        probabilities[probabilities < 1e-2] = 0.0

        forecast_data["class_probability"] = np.round(probabilities, 2)
        forecast_data["normalized_probability"] = forecast_data.groupby("Date")[
            "class_probability"
        ].transform(lambda x: x / x.sum())

        estimates.loc[estimates.index, "Daily Peak Probability"] = forecast_data.loc[
            estimates.index, "normalized_probability"
        ]

        logger.info("Saving the probability estimates")
        estimates = estimates.reset_index(drop=True)
        estimates["Date"] = estimates["Date"].dt.strftime("%m/%d/%Y")
        estimates.to_csv(
            data_dir / "results" / "probability_estimates.csv", index=False
        )
        return training_columns, estimates


if __name__ == "__main__":
    estimate_probability(test=False)
