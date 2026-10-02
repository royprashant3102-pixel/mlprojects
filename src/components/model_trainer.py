import os
import sys
from dataclasses import dataclass

# Gradient boosting libraries (install with: pip install catboost xgboost)
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# Scikit-learn regression models
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# Metric used to compare models
from sklearn.metrics import r2_score

# Project-level utilities
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    # Path where the best trained model will be saved
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        # Load config so we know where to save the trained model
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            # Split arrays into features (X) and target (y)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # all columns except last -> X_train (features)
                train_array[:, -1],   # only last column -> y_train (target)
                test_array[:, :-1],   # all columns except last -> X_test (features)
                test_array[:, -1],    # only last column -> y_test (target)
            )

            # Dictionary of all models we want to try
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter': ['best', 'random'],
                    # 'max_features': ['sqrt', 'log2'],
                },
                "Random Forest": {
                    # 'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features': ['sqrt', 'log2', None],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "Gradient Boosting": {
                    # 'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion': ['squared_error', 'friedman_mse'],
                    # 'max_features': ['auto', 'sqrt', 'log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression": {},  # no hyperparameters to tune
                "K-Neighbors Regressor": {
                    'n_neighbors': [5, 7, 9, 11],
                },
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100],
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    # 'loss': ['linear', 'square', 'exponential'],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
            }
            }

            # Train every model and get a report: {model_name: test_r2_score}
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict
            # Find the position of the best score, then pick the name at that position
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            # Get the actual trained model object
            best_model = models[best_model_name]

            # If even the best model scores below 0.6, treat it as no good model
            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)

            logging.info(
                f"Best model found: {best_model_name} with R2 score: {best_model_score}"
            )

            # Save the best model to artifacts/model.pkl
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            # Predict on test data with the best model and return its R2 score
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            # Wrap any error in our custom exception for detailed logging
            raise CustomException(e, sys)
