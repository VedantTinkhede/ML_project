import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'Linear Regression': LinearRegression(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(),
                'K-Neighbors': KNeighborsRegressor(),
                'XGBoost': XGBRegressor(),
                'CatBoost': CatBoostRegressor(verbose=0),
                'AdaBoost': AdaBoostRegressor(),
                'Gradient Boosting': GradientBoostingRegressor()
            }

            params = {
                'Linear Regression': {},

                'Decision Tree': {
                    'criterion': ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    'max_depth': [3, 5, 7, 9]
                },

                'Random Forest': {
                    'n_estimators': [10, 50, 100], 
                    'max_depth': [3, 5, 7, 9]
                },

                'K-Neighbors': {
                    'n_neighbors': [3, 5, 7, 10],
                    'weights': ['uniform', 'distance']
                },

                'XGBoost': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1]
                },

                'CatBoost': {
                    'iterations': [100, 200],
                    'depth': [3, 5]
                },

                'AdaBoost': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1]
                },

                'Gradient Boosting': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1]
                }
            }

            model_report:dict = evaluate_models(X_train = X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params) 

           
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy")

            logging.info(f"Best model found is: {best_model_name} with R2 score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)


            return r2_square

        except Exception as e:
            raise CustomException(e, sys)