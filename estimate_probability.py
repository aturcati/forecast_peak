from pathlib import Path

import pandas as pd
import pickle
import numpy as np
import warnings


import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV

from loguru import logger

import holidays

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def add_index(df):
    """Add index to the dataframe."""
    # Convert the 'Date' column to datetime format
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")

    # Combine 'Date' and 'Hour' into a single datetime column
    df["Datetime"] = df["Date"] + pd.to_timedelta(df["Hour"] - 1, unit="h")

    # Set the new datetime column as the index
    df.set_index("Datetime", inplace=True)
    return df


def expanding_window_cross_validation(
    df, initial_train_years=1, years_in_data=3, first_year=2005
):
    """Create expanding window cross-validation folds."""
    folds = []
    for year in range(initial_train_years, years_in_data):
        # Training set ends at the last hour of December 31st of the current year
        train_end_date = pd.Timestamp(f"{first_year + year}-12-31 23:00:00")

        # Testing set starts at the first hour of January 1st of the next year
        test_start_date = pd.Timestamp(f"{first_year + year + 1}-01-01 00:00:00")
        test_end_date = pd.Timestamp(f"{first_year + year + 1}-12-31 23:00:00")

        # Define the training and testing period
        train = df.loc[:train_end_date].copy()
        test = df.loc[test_start_date:test_end_date].copy()

        folds.append((train, test))

    return folds


def feature_engineering(df, holidays=None):
    """Feature engineering for the load data."""
    # Feature engineering
    logger.info("Feature engineering")

    # Temporal Features
    df["Dayofweek"] = df.index.dayofweek.astype("int")
    df["Dayofyear"] = df.index.dayofyear.astype("int")
    df["sin_Dayofyear"] = sin_encoding(df, "Dayofyear")
    df["Week"] = df.index.isocalendar().week.astype("int")
    df["Month"] = df.index.month.astype("int")
    df["Quarter"] = df.index.quarter.astype("int")
    df["Year"] = df.index.year.astype("int")
    if holidays is not None:
        df["Holiday"] = [1 if date in holidays else 0 for date in df.index.date]

    # Add template features
    week_profiles_peaks_train = template_single_column(
        df[["Month", "Dayofweek", "Hour", "is_peak"]],
        "is_peak",
        ["Month", "Dayofweek", "Hour"],
        methods=["mean", "std", "min", "max"],
        change_col_names=True,
    )

    temp_index = df.index.copy()
    df = df.merge(
        week_profiles_peaks_train, how="left", on=["Month", "Dayofweek", "Hour"]
    )
    df.index = temp_index

    # Add lagged features
    for station in [f"Station {i+1}" for i in range(28)]:
        for i in range(1, 4):
            df[f"{station}_lag_{i}"] = df[station].shift(i)
            df[f"{station}_lag_{i+11}"] = df[station].shift(i + 11)
            df[f"{station}_lag_{i+23}"] = df[station].shift(i + 23)

    return df.bfill()


def score(prediction, actual):
    """Calculate the score of a prediction."""
    return np.sum((prediction - actual) ** 2)


def sin_encoding(df, column):
    """Sinusoidal encoding of a column."""
    return np.sin(2 * np.pi * df[column] / df[column].max())


def template_single_column(
    dataframe: pd.DataFrame,
    column: str,
    grouping_columns: str,
    methods: str = ["mean"],
    change_col_names: bool = True,
    datetime_index: bool = False,
    index_format: str = None,
) -> pd.DataFrame:
    """Create template features for a single column."""
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


def tune_catboost(X_train, y_train):
    """Tune CatBoost model."""
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


def tune_lgbm(X_train, y_train):
    """Tune LightGBM model."""
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


def tune_xgboost(X_train, y_train):
    """Tune XGBoost model."""
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


if __name__ == "__main__":
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

    merged_data = feature_engineering(merged_data, holidays=us_holidays)

    # Drop the columns that are not needed
    training_data = merged_data.drop(columns=["Date", "Load", "Dayofyear"])

    # Define the target and training columns
    target = "is_peak"
    training_columns = training_data.columns.drop(target)

    logger.info(f"Target variable: {target}")
    logger.info("Training variables:")
    for i, c in enumerate(training_columns):
        logger.info(f"{i+1}) {c}")

    # Expanding window cross-validation
    folds = expanding_window_cross_validation(
        training_data, initial_train_years=0, years_in_data=3, first_year=2005
    )

    test = True
    if test:
        logger.info("Testing the model")
        # Expanding window cross-validation
        for i, (train, test) in enumerate(folds[:-1]):
            X_train, y_train = train[training_columns], train[target]
            X_test, y_test = test[training_columns], test[target]

            # 1. XGBoost Classifier
            logger.info("Tuning XGBoost model")
            xgb_clf = tune_xgboost(X_train, y_train)

            # 2. LightGBM Classifier
            logger.info("Tuning LightGBM model")
            lgb_clf = tune_lgbm(X_train, y_train)

            # 3. CatBoost Classifier
            logger.info("Tuning CatBoost model")
            cat_clf = tune_catboost(X_train, y_train)

            ensemble_clf = VotingClassifier(
                estimators=[("xgb", xgb_clf), ("lgb", lgb_clf), ("cat", cat_clf)],
                voting="soft",
            )

            # Calibrate the model
            cal_clf = CalibratedClassifierCV(
                ensemble_clf, method="isotonic", ensemble=True
            )

            # Train the model
            cal_clf.fit(X_train, y_train)

            # Get probabilistic predictions (probabilities for each class)
            probabilities = cal_clf.predict_proba(X_test)

            test["Date"] = test.index.date
            test["class_probability"] = probabilities[:, 1]
            test["normalized_probability"] = test.groupby("Date")[
                "class_probability"
            ].transform(lambda x: x / x.sum())
            test["predicted_is_peak"] = cal_clf.predict(X_test)

            test.to_csv(data_dir / "results" / f"fold_{i+1}_results.csv")

            logger.info(
                f"Fold {i+1}: Score {score(test["normalized_probability"], y_test)}"
            )
    else:
        train, _ = folds[-1]
        X_train, y_train = train[training_columns], train[target]

        # 1. XGBoost Classifier
        logger.info("Tuning XGBoost model")
        xgb_clf = tune_xgboost(X_train, y_train)

        # 2. LightGBM Classifier
        logger.info("Tuning LightGBM model")
        lgb_clf = tune_lgbm(X_train, y_train)

        # 3. CatBoost Classifier
        logger.info("Tuning CatBoost model")
        cat_clf = tune_catboost(X_train, y_train)

        ensemble_clf = VotingClassifier(
            estimators=[("xgb", xgb_clf), ("lgb", lgb_clf), ("cat", cat_clf)],
            voting="soft",
        )

        # Calibrate the model
        cal_clf = CalibratedClassifierCV(ensemble_clf, method="isotonic", ensemble=True)

        # Train the model
        cal_clf.fit(X_train, y_train)

        # Saving the model to disk
        with open(data_dir / "models" / "model.pkl", "wb") as f:
            pickle.dump(cal_clf, f)

        # Get probabilistic predictions (probabilities for each class)
        probabilities = cal_clf.predict_proba()
