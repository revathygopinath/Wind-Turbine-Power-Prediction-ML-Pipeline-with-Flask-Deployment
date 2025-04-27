from dataclasses import dataclass
import logging
import os
from sklearn.linear_model import LinearRegression
import sys
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            # Initialize RandomizedSearchCV for each model
            random_search_results = {}

            for model_name, model in models.items():
                logging.info(f"Running RandomizedSearchCV for {model_name}")

                # Get the hyperparameter grid for the model
                model_params = params.get(model_name, {})

                # If there are no parameters for the model, skip RandomizedSearchCV
                if model_params:
                    random_search = RandomizedSearchCV(estimator=model,
                                                       param_distributions=model_params,
                                                       n_iter=50,  # Number of random combinations to try
                                                       cv=5,  # 5-fold cross-validation
                                                       verbose=2,  # Show progress
                                                       n_jobs=-1,  # Use all CPU cores
                                                       random_state=42)

                    random_search.fit(X_train, y_train)

                    # Store the best score and best model
                    random_search_results[model_name] = {
                        'best_model': random_search.best_estimator_,
                        'best_score': random_search.best_score_
                    }

            # Get the best model based on performance
            best_model_name = max(random_search_results, key=lambda x: random_search_results[x]['best_score'])
            best_model = random_search_results[best_model_name]['best_model']

            best_model_score = random_search_results[best_model_name]['best_score']
            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient score")

            logging.info(f"Best found model: {best_model_name} with score: {best_model_score}")

            # Save the best model
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            # Make predictions with the best model
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)