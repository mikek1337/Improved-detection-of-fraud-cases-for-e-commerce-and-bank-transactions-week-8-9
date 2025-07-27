import pandas as pd
import numpy as np
from typing import Callable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# --- 1. Custom Transformer for Time-Based and Frequency/Velocity Features ---
class CustomFeatureEngineerForFraud(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn transformer for engineering time-based, frequency,
    and velocity features from transactional data.

    This transformer calculates:
    - `time_since_signup`: The duration in seconds between a user's signup and purchase.
    - `hour_of_day`: The hour at which a purchase was made.
    - `day_of_week`: The day of the week a purchase was made.
    - `transaction_frequency`: The total number of transactions for each user.
    - `avg_time_between_transactions_seconds`: The average time difference in seconds
      between consecutive transactions for each user.

    Attributes:
        avg_time_between_transactions_ (pd.Series): Stores the average time between
            transactions for each user, learned during the fit method.
        user_transaction_frequency_ (pd.Series): Stores the frequency of transactions
            for each user, learned during the fit method.
    """
    def __init__(self):
        """
        Initializes the CustomFeatureEngineer.
        No parameters are required at initialization.
        """
        pass

    def fit(self, X, y=None):
        """
        Fits the transformer by calculating user-specific statistics from the training data.

        This method calculates:
        - The average time between transactions for each user.
        - The total transaction frequency for each user.

        These statistics are stored as attributes to be used during the transform phase
        on both training and test datasets.

        Args:
            X (pd.DataFrame): The input DataFrame containing 'user_id', 'purchase_time',
                              and 'signup_time' columns.
            y (pd.Series, optional): Ignored. Present for scikit-learn API compatibility.

        Returns:
            self: The fitted transformer instance.
        """
        # We need to calculate the average time between transactions based on the training data's user_ids
        # to handle cases where users only have one transaction in the training set
        temp_df = X.copy()
        temp_df['purchase_time'] = pd.to_datetime(temp_df['purchase_time'])
        temp_df['signup_time'] = pd.to_datetime(temp_df['signup_time'])

        # Sort for time_diff calculation
        temp_df_sorted = temp_df.sort_values(by=['user_id', 'purchase_time'])
        temp_df_sorted['time_diff'] = temp_df_sorted.groupby('user_id')['purchase_time'].diff()
        temp_df_sorted['time_diff_seconds'] = temp_df_sorted['time_diff'].dt.total_seconds()

        # Calculate average time between transactions for each user in the training set
        self.avg_time_between_transactions_ = temp_df_sorted.groupby('user_id')['time_diff_seconds'].mean()
        # Fill NaN for users with single transaction (no diff to calculate) - typically 0 or a placeholder
        self.avg_time_between_transactions_ = self.avg_time_between_transactions_.fillna(0) # Or another suitable value

        # Calculate transaction frequency per user based on training data
        self.user_transaction_frequency_ = temp_df['user_id'].value_counts()
        return self

    def transform(self, X, y=None):
        """
        Transforms the input DataFrame by adding engineered features.

        This method applies the calculations for time-based features and maps
        the user-specific statistics (frequency and average time between transactions)
        learned during the `fit` phase to the input data.

        Args:
            X (pd.DataFrame): The input DataFrame containing 'user_id', 'purchase_time',
                              and 'signup_time' columns.
            y (pd.Series, optional): Ignored. Present for scikit-learn API compatibility.

        Returns:
            pd.DataFrame: The DataFrame with new engineered features added.
        """
        X_transformed = X.copy()

        # Ensure timestamp columns are datetime objects
        # X_transformed['timestamp'] = pd.to_datetime(X_transformed['timestamp'])
        X_transformed['purchase_time'] = pd.to_datetime(X_transformed['purchase_time'])
        X_transformed['signup_time'] = pd.to_datetime(X_transformed['signup_time'])

        # Feature: time_since_signup
        X_transformed['time_since_signup'] = X_transformed['purchase_time'] - X_transformed['signup_time']
        X_transformed['time_since_signup'] = X_transformed['time_since_signup'].dt.total_seconds()

        # Feature: hour_of_day
        X_transformed['hour_of_day'] = X_transformed['purchase_time'].dt.hour

        # Feature: day_of_week
        X_transformed['day_of_week'] = X_transformed['purchase_time'].dt.dayofweek

        # Feature: transaction_frequency (per user)
        # Use a map to apply the frequencies learned during fit
        X_transformed['transaction_frequency'] = X_transformed['user_id'].map(self.user_transaction_frequency_).fillna(0)

        # Feature: avg_time_between_transactions_seconds (per user)
        # Handle cases where a user in test set might not be in training set (fillna with 0 or mean)
        X_transformed['avg_time_between_transactions_seconds'] = X_transformed['user_id'].map(self.avg_time_between_transactions_).fillna(0)


        # Drop original time columns if they are not needed for the model
        # X_transformed = X_transformed.drop(columns=['purchase_time', 'signup_time', 'user_id', 'device_id'], errors='ignore')

        return X_transformed



def process_frude(fraud_data:pd.DataFrame, numerical_cols, categorical_cols, pipeline:Callable[[list[str],list[str]], Pipeline], target_col:str='class'):
    """
    Processes the fraud detection dataset by performing feature engineering,
    data splitting, preprocessing (scaling and one-hot encoding), and
    handling class imbalance using SMOTE.

    Args:
        fraud_data (pd.DataFrame): The input DataFrame containing the fraud data.
                                   Expected to have 'purchase_time', 'signup_time',
                                   'user_id', 'class' and other numerical/categorical columns.
        numerical_cols (list): A list of column names that are numerical features.
        categorical_cols (list): A list of column names that are categorical features.

    Returns:
        dict: A dictionary containing the processed training and testing datasets:
              - 'x_train': Processed training features (pd.DataFrame).
              - 'y_train': Training target labels (pd.Series), resampled by SMOTE.
              - 'x_test': Processed testing features (pd.DataFrame).
    """
    # --- 3. Define Features (X) and Target (y) ---
    # Make sure to include all columns that will be processed by the pipeline
    # Keep original time columns for CustomFeatureEngineer to process
    feature_cols = numerical_cols + categorical_cols
    print(feature_cols)


    X = fraud_data[feature_cols]
    y = fraud_data[target_col]

    # Identify categorical and numerical columns that will be processed by StandardScaler/OneHotEncoder
    # These are the *final* types of features after CustomFeatureEngineer
    final_numerical_cols = numerical_cols
    final_categorical_cols = categorical_cols

    # --- 4. Split Data into Training and Testing Sets ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print("Original Training Set Class Distribution:")
    print(y_train.value_counts())
    print("-" * 50)

    # --- 5. Create the Preprocessing Pipeline ---

    # Define the preprocessor using ColumnTransformer for numerical and categorical features
    feature_engineering_pipeline = pipeline(final_numerical_cols, final_categorical_cols)

    # --- 6. Apply the Pipeline to Training Data ---
    X_train_processed = feature_engineering_pipeline.fit_transform(X_train, y_train)

    # --- 7. Apply the Pipeline to Test Data (using fitted transformers from training) ---
    X_test_processed = feature_engineering_pipeline.transform(X_test)

    # Convert processed arrays back to DataFrame for better inspection (optional but good for debugging)
    # Get feature names after one-hot encoding
    cat_feature_names = feature_engineering_pipeline.named_steps['preprocessor'].get_feature_names_out()
    all_feature_names = cat_feature_names # Combine numerical and one-hot encoded names
    # If remainder='passthrough', user_id might be added here too, handle if needed.
    # For simplicity, assuming user_id is dropped or not part of the final X for modeling.

    X_train_processed_df = pd.DataFrame(X_train_processed, columns=all_feature_names, index=y_train.index)
    X_test_processed_df = pd.DataFrame(X_test_processed, columns=all_feature_names, index=y_test.index)

    print("Shape of X_train_processed_df after pipeline:", X_train_processed_df.shape)
    print("First 5 rows of X_train_processed_df (features engineered, scaled, encoded):")
    print(X_train_processed_df.head())
    print("-" * 50)

    print("Shape of X_test_processed_df after pipeline:", X_test_processed_df.shape)
    print("First 5 rows of X_test_processed_df (features engineered, scaled, encoded):")
    print(X_test_processed_df.head())
    print("-" * 50)

    # --- 8. Handle Class Imbalance with SMOTE (on training data only, after preprocessing) ---
    # SMOTE requires numerical input, which is why it comes after encoding and scaling.
    print("Class Distribution in Training Set BEFORE SMOTE (after preprocessing):")
    print(y_train.value_counts())
    print("-" * 50)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed_df, y_train)

    print("Class Distribution in Training Set AFTER SMOTE:")
    print(y_train_resampled.value_counts())
    print(y_train_resampled.value_counts(normalize=True) * 100)
    print("-" * 50)
    print("\nFeature Engineering Pipeline successfully created and applied. Data is ready for model training. ✨")
    return {'x_train':X_train_resampled, 'y_train': y_train_resampled, 'x_test':X_test_processed, 'y_test': y_test}

def fraud_pipeline(final_numerical_cols:list[str], final_categorical_cols:list[str]):
    """
    Creates and returns a scikit-learn Pipeline specifically designed for fraud detection
    feature engineering and preprocessing.

    This pipeline includes:
    1. A custom feature engineering step (`CustomFeatureEngineerForFraud`) to derive
       domain-specific features (e.g., time-based, frequency).
    2. A `ColumnTransformer` for standard preprocessing tasks:
       - `StandardScaler` for numerical features.
       - `MinMaxScaler` for numerical features (applied to the same set, which might be
         redundant or intended for specific features if `final_numerical_cols` is split).
       - `OneHotEncoder` for categorical features.

    Args:
        final_numerical_cols (list[str]): A list of column names that should be treated
                                          as numerical features for scaling.
        final_categorical_cols (list[str]): A list of column names that should be treated
                                            as categorical features for one-hot encoding.

    Returns:
        Pipeline: A scikit-learn Pipeline object ready to be fitted and transformed.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), final_numerical_cols),
            ('minmax', MinMaxScaler(), final_numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=7), final_categorical_cols)
        ],
        remainder='passthrough' # Keep other columns (like 'user_id' if not used in model)
    )

    # Combine CustomFeatureEngineer and the preprocessor into a main pipeline
    feature_engineering_pipeline = Pipeline(steps=[
        ('custom_features', CustomFeatureEngineerForFraud()),
        ('preprocessor', preprocessor)
    ])

    return feature_engineering_pipeline

def credit_pipeline(final_numerical_cols:list[str],final_categorical_cols:list[str]):
    """
    Creates and returns a scikit-learn Pipeline for credit data preprocessing.

    This pipeline is simpler than the fraud pipeline as it does not include a
    custom feature engineering step. It focuses solely on standard preprocessing:
    - `StandardScaler` for numerical features.
    - `OneHotEncoder` for categorical features.

    Args:
        final_numerical_cols (list[str]): A list of column names that should be treated
                                          as numerical features for scaling.
        final_categorical_cols (list[str]): A list of column names that should be treated
                                            as categorical features for one-hot encoding.

    Returns:
        Pipeline: A scikit-learn Pipeline object ready to be fitted and transformed.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), final_numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=7), final_categorical_cols)
        ],
        remainder='passthrough' # Keep other columns (like 'user_id' if not used in model)
    )

    # Combine CustomFeatureEngineer and the preprocessor into a main pipeline
    feature_engineering_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])

    return feature_engineering_pipeline

def test_featureengineering(numerical_cols:list[str], categorical_cols:list[str]):
    """
    A utility function to create a fraud detection feature engineering pipeline,
    primarily intended for testing or demonstration purposes.

    This function simply calls `fraud_pipeline` with the provided numerical and
    categorical column lists.

    Args:
        numerical_cols (list[str]): A list of column names that are numerical features.
        categorical_cols (list[str]): A list of column names that are categorical features.

    Returns:
        Pipeline: A scikit-learn Pipeline object configured for fraud detection
                  feature engineering and preprocessing.
    """
    pipeline = fraud_pipeline(numerical_cols,categorical_cols)
    return pipeline
