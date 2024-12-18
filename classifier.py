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
from aif360.metrics import ClassificationMetric
from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.datasets import BinaryLabelDataset
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix

 # Ensure TensorFlow compatibility with AIF360
tf.compat.v1.disable_eager_execution()

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
    def __init__(self, full_data):
        """
        Initialize classification pipeline
        Args:
            full_data: Complete dataset
        """
        self.full_data = full_data
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
        
        # 1. Binarize age feature (available for other project parts)
        age_median = df['age'].median()
        df['age_binary'] = (df['age'] > age_median).astype(int)
        df = df.drop('age', axis=1) # Remove original age column
        
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
        Prepare and split data after preprocessing
        """
        # First preprocess all data
        X_full = self.preprocess_features(self.full_data)
        y_full = self.full_data['income_class']
        
        # Now split into train/val/test
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_full, y_full, 
            test_size=0.2,
            random_state=42,
            stratify=y_full
        )

        # Second split: create validation set
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=0.25,
            random_state=42,
            stratify=y_temp
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

        self.evaluate_performance(y,predictions,dataset_name)
        
        # Plot feature importance
        if dataset_name == "Test":
            self.plot_feature_importance()
        
        return predictions
    
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

    def evaluate_fairness(self,test_data,predicted_dataset,privileged_groups,unprivileged_groups,label):
        """
        Evaluate fairnesse metrics based on the privileged groups, true label and the predicted label
        """
        # Evaluate fairness metrics
        metric = ClassificationMetric(
            test_data,
            predicted_dataset,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups
        )
        print(f"Disparate Impact {label}: {metric.disparate_impact()}")
        print(f"Statistical Parity Difference {label}: {metric.statistical_parity_difference()}")


    def evaluate_performance(self,y_true, y_pred,label):
        """
        Evaluate performance of a given model with multiple metrics
        """
        print(f"\n{label} Performance: ")
        print("-" * 50)

        # Calculate performance metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        # Print metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))

        # Return a dictionary of performance metrics
        performance_metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
        return performance_metrics
    
    def adversarial_debiasing(self,train_data,test_data,privileged_groups,unprivileged_groups):
        """
        Evaluate performance of a given model with multiple metrics
        """
        debiased_model = AdversarialDebiasing(
            privileged_groups=privileged_groups,
            unprivileged_groups=unprivileged_groups,
            scope_name='adversarial_debiasing',
            debias=True,  # Enable debiasing
            sess=tf.compat.v1.Session()
        )

        # Train the model
        debiased_model.fit(train_data)

        # Evaluate on test data
        predicted_test_dataset = debiased_model.predict(test_data)

        # Convert AIF360 dataset back to numpy arrays for performance metrics
        y_pred_adversarial_debiasing = predicted_test_dataset.labels.flatten()

        return y_pred_adversarial_debiasing,predicted_test_dataset

class ClassificationVisualizer:
    def __init__(self, pipeline, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Initialize visualizer with data and pipeline
        """
        self.pipeline = pipeline
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        # Create Pictures directory if it doesn't exist
        if not os.path.exists('Pictures'):
            os.makedirs('Pictures')
            
        # Set style for all plots
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def set_common_style(self):
        """
        Set common style elements for all plots
        """
        plt.rcParams['figure.figsize'] = [10, 6]
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
    
    def plot_data_split(self):
        """
        Visualize the data split proportions
        """
        self.set_common_style()
        
        # Calculate sizes
        sizes = [len(self.X_train), len(self.X_val), len(self.X_test)]
        labels = ['Training Set\n(60%)', 'Validation Set\n(20%)', 'Test Set\n(20%)']
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        plt.figure(figsize=(10, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, pctdistance=0.85)
        plt.title('Dataset Split Distribution', pad=20)
        
        # Draw center circle
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        
        plt.tight_layout()
        plt.savefig('Pictures/data_split.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_confusion_matrix(self):
        """
        Create and plot confusion matrix
        """
        self.set_common_style()
        
        # Get predictions
        y_pred = self.pipeline.model.predict(self.X_test)
        
        # Calculate confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['≤50K', '>50K'],
                    yticklabels=['≤50K', '>50K'])
        
        plt.title('Confusion Matrix', pad=20)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        plt.tight_layout()
        plt.savefig('Pictures/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_feature_importance(self):
        """
        Plot feature importance from the model
        """
        self.set_common_style()
        
        # Get feature importance
        importance = self.pipeline.model.feature_importances_
        features = sorted(zip(self.pipeline.feature_names, importance),
                        key=lambda x: x[1], reverse=True)[:20]
        
        # Prepare data for plotting
        feature_names = [f[0] for f in features]
        importance_values = [f[1] for f in features]
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(feature_names)), importance_values)
        
        # Add value labels on the bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', 
                    ha='left', va='center', fontsize=10)
        
        plt.yticks(range(len(feature_names)), feature_names)
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Most Important Features', pad=20)
        
        plt.tight_layout()
        plt.savefig('Pictures/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_class_distribution(self):
        """
        Visualize class distribution in the dataset
        """
        self.set_common_style()
        
        # Calculate class distributions
        train_dist = np.bincount(self.y_train) / len(self.y_train)
        val_dist = np.bincount(self.y_val) / len(self.y_val)
        test_dist = np.bincount(self.y_test) / len(self.y_test)
        
        # Set up the plot
        x = np.arange(2)
        width = 0.25
        
        plt.figure(figsize=(10, 6))
        plt.bar(x - width, train_dist, width, label='Training Set')
        plt.bar(x, val_dist, width, label='Validation Set')
        plt.bar(x + width, test_dist, width, label='Test Set')
        
        plt.xticks(x, ['≤50K', '>50K'])
        plt.ylabel('Proportion of Samples')
        plt.title('Class Distribution Across Dataset Splits', pad=20)
        plt.legend()
        
        # Add percentage labels on bars
        def add_labels(dist, offset):
            for i, v in enumerate(dist):
                plt.text(i + offset, v, f'{v*100:.1f}%',
                        ha='center', va='bottom')
        
        add_labels(train_dist, -width)
        add_labels(val_dist, 0)
        add_labels(test_dist, width)
        
        plt.tight_layout()
        plt.savefig('Pictures/class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_all_visualizations(self):
        """
        Generate all visualizations
        """
        print("Generating visualizations...")
        self.plot_data_split()
        self.plot_confusion_matrix()
        self.plot_feature_importance()
        self.plot_class_distribution()
        print("All visualizations have been saved in the Pictures directory.")

def main():
    """
    Main execution function that runs the complete pipeline
    """
    print("Starting Adult Income Classification Pipeline...")
    
    # Read and clean data
    print("\nReading and cleaning dataset...")
    full_data = read_dataset()
    cleaned_data = clean_dataset(full_data)
    
    # Initialize pipeline with full dataset
    pipeline = AdultClassificationPipeline(cleaned_data)
    
    # Get splits into train, test and validation data
    print("\nPreparing features...")
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.prepare_data()

    # Train model
    pipeline.train_model(X_train, X_val, y_train, y_val)
    
    # Evaluate model
    pipeline.evaluate_model(X_val, y_val, "Validation")
    y_pred = pipeline.evaluate_model(X_test, y_test, "Test")

    print("\nEvaluating fairness...")
    # Convert the data into a BinaryLabelDataset (AIF360 format)
    train_data = BinaryLabelDataset(df=pd.concat([X_train, y_train], axis=1), label_names=['income_class'], protected_attribute_names=['sex_Male', 'age_binary'])
    test_data = BinaryLabelDataset(df=pd.concat([X_test, y_test], axis=1), label_names=['income_class'], protected_attribute_names=['sex_Male', 'age_binary'])

    # Define privileged and unprivileged groups (AIF360)
    privileged_groups = [{'sex_Male': 1}]
    unprivileged_groups = [{'sex_Male': 0}]

    # Use the same features and protected attributes from test_data (AIF360)
    predicted_dataset = test_data.copy(deepcopy=True)

    # Replace the labels with the predicted values (AIF360)
    predicted_dataset.labels = y_pred.reshape(-1, 1)  # Ensure y_pred has the correct shape

    # Evaluate fairness for original model
    pipeline.evaluate_fairness(test_data,predicted_dataset,privileged_groups,unprivileged_groups,'before adversarial debiasing')

    print("\nDebiasing Model...")
    # Adversarial loss function that penalizes the model if it uses sensitive features (sex) to make predictions
    y_pred_adversarial_debiasing,predicted_test_dataset = pipeline.adversarial_debiasing(train_data,test_data,privileged_groups,unprivileged_groups)
    # Evaluate new model
    pipeline.evaluate_performance(y_test,y_pred_adversarial_debiasing,"Test - Adversarial Debiasing")
    # Evaluate fairness for new model
    pipeline.evaluate_fairness(test_data,predicted_test_dataset,privileged_groups,unprivileged_groups,'after adversarial debiasing')

    # Create visualizations
    visualizer = ClassificationVisualizer(
        pipeline=pipeline,
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test
    )
    visualizer.create_all_visualizations()
    
    return pipeline

if __name__ == "__main__":
    pipeline = main()