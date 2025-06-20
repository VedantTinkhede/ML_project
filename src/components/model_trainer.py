import os
import sys
from dataclasses import dataclass

from sklearn.multioutput import MultiOutputRegressor
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
            X_train, y_train = train_array[:, :-3], train_array[:, -3:]
            X_test, y_test = test_array[:, :-3], test_array[:, -3:]

            print("X_train shape:", X_train.shape)
            print("y_train shape:", y_train.shape)

            models = {
                'Linear Regression': MultiOutputRegressor(LinearRegression()),
                'Decision Tree': MultiOutputRegressor(DecisionTreeRegressor()),
                'Random Forest': MultiOutputRegressor(RandomForestRegressor()),
                'K-Neighbors': MultiOutputRegressor(KNeighborsRegressor()),
                'XGBoost': MultiOutputRegressor(XGBRegressor()),
                'CatBoost': MultiOutputRegressor(CatBoostRegressor(verbose=0)),
                'AdaBoost': MultiOutputRegressor(AdaBoostRegressor()),
                'Gradient Boosting': MultiOutputRegressor(GradientBoostingRegressor())
            }

            params = {
                'Linear Regression': {},

                'Decision Tree': {
                    'estimator__criterion': ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    'estimator__max_depth': [3, 5, 7, 9]
                },

                'Random Forest': {
                    'estimator__n_estimators': [10, 50, 100], 
                    'estimator__max_depth': [3, 5, 7, 9]
                },

                'K-Neighbors': {
                    'estimator__n_neighbors': [3, 5, 7, 10],
                    'estimator__weights': ['uniform', 'distance']
                },

                'XGBoost': {
                    'estimator__n_estimators': [50, 100],
                    'estimator__learning_rate': [0.01, 0.1]
                },

                'CatBoost': {
                    'estimator__iterations': [100, 200],
                    'estimator__depth': [3, 5]
                },

                'AdaBoost': {
                    'estimator__n_estimators': [50, 100],
                    'estimator__learning_rate': [0.01, 0.1]
                },

                'Gradient Boosting': {
                    'estimator__n_estimators': [50, 100],
                    'estimator__learning_rate': [0.01, 0.1]
                }
            }

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )

            print("Model scores:")
            for model_name, score in model_report.items():
                print(f"{model_name}: R2 = {score}")


            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            best_model.fit(X_train, y_train)

            # if best_model_score < 0.6:
            #     raise CustomException("No best model found with sufficient accuracy", sys)

            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
