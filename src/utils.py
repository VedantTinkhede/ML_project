import os
import sys
import numpy as np
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict):
    """
    Evaluates given models using GridSearchCV and returns a dictionary of model names with their average R² scores.
    Assumes models are already wrapped in MultiOutputRegressor (if needed).
    """
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Evaluating: {model_name}")
            model_params = params.get(model_name, {})

            # GridSearchCV on the provided model (already wrapped in MultiOutputRegressor if needed)
            gs = GridSearchCV(estimator=model, param_grid=model_params, cv=3, n_jobs=-1, verbose=0)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_
            y_pred = best_model.predict(X_test)

            # Average R² over multiple outputs
            r2_scores = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])]
            avg_r2 = np.mean(r2_scores)

            report[model_name] = avg_r2

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
