import os
import sys
from dataclasses import dataclass # used to create configuration classes
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer # used to create pipelines for preprocessing
from sklearn.impute import SimpleImputer # used to handle missing values
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object  # utility function to save the preprocessor object as a pickle file

@dataclass 
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')  # save model in artifacts folder as a pickle file

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation.
        It creates a preprocessing pipeline for numerical and categorical features.
        """
        try:
            numerical_features = ['writing_score', 'reading_score']
            categorical_features = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

            # Numerical pipeline
            numerical_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='median')),  # Impute missing values with median
                ('scaler', StandardScaler(with_mean=False))  # Scale numerical features
                ]
            )

            # Categorical pipeline
            categorical_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with most frequent
                ('one_hot_encoder', OneHotEncoder()),  # Convert categorical variables into dummy/indicator variables
                ('scaler', StandardScaler(with_mean=False))  # Scale categorical features
                ]
            )

            logging.info("Numerical and categorical pipelines created successfully")

            # Combine both pipelines into a ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('numerical_pipeline', numerical_pipeline, numerical_features),   # numerical_pipeline and categorical_pipeline are the names of the pipelines
                    ('categorical_pipeline', categorical_pipeline, categorical_features)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        """
        This function is responsible for initiating the data transformation process.
        It reads the training and testing data, applies the preprocessing pipeline,
        and saves the transformed data to a pickle file.
        """
        try:
            train_df = pd.read_csv(train_path)  # Read training data
            test_df = pd.read_csv(test_path)    # Read testing data

            logging.info("Data read successfully from CSV files")

            preprocessor_obj = self.get_data_transformer_object()  # Get the preprocessing pipeline

            target_column_name = "math_score"  # Specify the target column

            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)  # Drop target column from training data
            target_feature_train_df = train_df[target_column_name]  # Extract target column from training data

            input_features_test_df = test_df.drop(columns=[target_column_name], axis=1)  # Drop target column from testing data
            target_feature_test_df = test_df[target_column_name]  # Extract target column from testing data

            logging.info("Applying transformations on training and testing data")

            # Fit and transform the training data, transform the testing data
            input_feature_train_arr = preprocessor_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_features_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]  # Combine input features and target for training data
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]  # Combine input features and target for testing data

            logging.info("Transformations applied successfully")

            # Save the preprocessor object to a file
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj= preprocessor_obj
            )

            return (
                train_arr, 
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path 
                
            )  # Return transformed training data, preprocessor file path, and transformed testing data
        
        except Exception as e:
            raise CustomException(e, sys)
            