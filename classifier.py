import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

# Path Configuration
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(CURRENT_DIR, 'adult.csv')

# Updated data types based on the actual data format
data_types = {
    "age": np.float64,
    "workclass": "category",
    "fnlwgt": np.float64,
    "education": "category",
    "educational-num": np.float64,
    "marital-status": "category",
    "occupation": "category",
    "relationship": "category",
    "race": "category",
    "gender": "category",
    "capital-gain": np.float64,
    "capital-loss": np.float64,
    "hours-per-week": np.float64,
    "native-country": "category",
    "income": "category"
}

def read_dataset():
    """
    Read the adult.csv dataset with appropriate data types and handling
    """
    print(f"Reading data from: {DATA_FILE}")
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file not found at: {DATA_FILE}")
    
    df = pd.read_csv(
        DATA_FILE,
        dtype=data_types,
        na_values=['?'],
    )
    
    # Rename columns to standardized format
    df = df.rename(columns={
        'educational-num': 'education_num',
        'marital-status': 'marital_status',
        'capital-gain': 'capital_gain',
        'capital-loss': 'capital_loss',
        'hours-per-week': 'hours_per_week',
        'native-country': 'native_country',
        'gender': 'sex'  # Rename gender to sex for consistency
    })
    
    print("\nDataset shape:", df.shape)
    print("\nColumns after renaming:", df.columns.tolist())
    print("\nMissing values:\n", df.isnull().sum())
    
    return df

def clean_dataset(data):
    """
    Clean the dataset by handling missing values and preparing target variable
    """
    # Make a copy to avoid modifying the original data
    data = data.copy()
    
    # Remove fnlwgt column
    if 'fnlwgt' in data.columns:
        data = data.drop('fnlwgt', axis=1)
    
    # Remove duplicates
    data = data.drop_duplicates()
    
    # Binarize target variable (>50K == 1 and <=50K == 0)
    data['income'] = data['income'].apply(lambda x: 1 if '>50K' in str(x) else 0)
    data = data.rename(columns={'income': 'income_class'})
    
    # Fill missing values
    categorical_columns = data.select_dtypes(include=['category']).columns
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    
    # Fill missing values for categorical columns with mode
    for col in categorical_columns:
        data[col] = data[col].fillna(data[col].mode()[0])
    
    # Fill missing values for numerical columns with median
    for col in numerical_columns:
        data[col] = data[col].fillna(data[col].median())
    
    return data

class AdultClassificationPipeline:
    def __init__(self, train_data, test_data):
        """
        Initialize classification pipeline
        Args:
            train_data: Training dataset
            test_data: Test dataset
        """
        self.train_data = train_data
        self.test_data = test_data
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def preprocess_features(self, df):
        """
        Basic feature preprocessing pipeline as required by the project:
        - Age binarization
        - Basic numerical feature standardization
        - Basic categorical feature encoding
        """
        df = df.copy()
        
        # 1. Binarize age feature
        age_median = df['age'].median()
        df['age_binary'] = (df['age'] > age_median).astype(int)
        df = df.drop('age', axis=1)  # Remove original age column
        
        # 2. Feature splitting
        numeric_features = ['education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
        categorical_features = ['workclass', 'education', 'marital_status', 'occupation',
                            'relationship', 'race', 'sex', 'native_country']
        
        # 3. Standardization of numerical features
        df[numeric_features] = self.scaler.fit_transform(df[numeric_features])
        
        # 4. Categorical encoding
        df_categorical = pd.get_dummies(df[categorical_features])
        
        # 5. Combine features
        features = pd.concat([
            df[numeric_features],
            df_categorical,
            df[['age_binary']]
        ], axis=1)
        
        self.feature_names = features.columns.tolist()
        return features
    
    def prepare_data(self):
        """
        Prepare train, validation and test data
        """
        # Prepare features
        X_full = self.preprocess_features(self.train_data)
        y_full = self.train_data['income_class']
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_full, y_full, 
            test_size=0.2, 
            random_state=42,
            stratify=y_full  # Ensure balanced splits
        )
        
        # Second split: create validation set from remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=0.25,
            random_state=42,
            stratify=y_temp  # Ensure balanced splits
        )
        
        print("\nData split sizes:")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_model(self, X_train, X_val, y_train, y_val):
        """
        Train LightGBM model with validation-based early stopping
        """
        print("\nTraining LightGBM model...")
        
        # Define model with balanced class weights
        self.model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=32,
            max_depth=6,
            class_weight='balanced',
            random_state=42
        )
        
        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[
                early_stopping(stopping_rounds=50),
                log_evaluation(period=100)
            ]
        )
        
        return self.model
    
    def evaluate_model(self, X, y, dataset_name=""):
        """
        Evaluate model performance
        """
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]

        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions)
        recall = recall_score(y, predictions)
        f1 = f1_score(y, predictions)
        roc_auc = roc_auc_score(y, probabilities)
        
        print(f"\n{dataset_name} Set Performance:")
        print("-" * 50)
        print(f"Accuracy: {accuracy_score(y, predictions):.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y, predictions))
        
        # Plot feature importance
        if dataset_name == "Test":
            self.plot_feature_importance()
        
        return predictions, probabilities
    
    def plot_feature_importance(self):
        """
        Plot feature importance from the trained model
        """
        importance = self.model.feature_importances_
        features = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        })
        features = features.sort_values('Importance', ascending=False).head(20)
        
        plt.figure(figsize=(12, 6))
        plt.title("Top 20 Feature Importances")
        plt.barh(range(len(features)), features['Importance'])
        plt.yticks(range(len(features)), features['Feature'])
        plt.tight_layout()
        plt.show()

def main():
    """
    Main execution function that runs the complete pipeline
    """
    print("Starting Adult Income Classification Pipeline...")
    
    # Read and clean data
    print("\nReading and cleaning dataset...")
    full_data = read_dataset()
    cleaned_data = clean_dataset(full_data)
    
    # Split into train and test
    print("\nSplitting into train and test sets...")
    train_data, test_data = train_test_split(
        cleaned_data, 
        test_size=0.2, 
        random_state=42,
        stratify=cleaned_data['income_class']  # Ensure balanced splits
    )
    
    # Initialize pipeline
    pipeline = AdultClassificationPipeline(train_data, test_data)
    
    # Prepare data
    print("\nPreparing features...")
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.prepare_data()
    
    # Train model
    pipeline.train_model(X_train, X_val, y_train, y_val)
    
    # Evaluate model
    pipeline.evaluate_model(X_val, y_val, "Validation")
    pipeline.evaluate_model(X_test, y_test, "Test")
    
    return pipeline

if __name__ == "__main__":
    pipeline = main()