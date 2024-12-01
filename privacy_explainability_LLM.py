import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from classifier_with_privacy_fairness import AdultClassificationPipeline, PrivateDatasetPipeline, read_dataset, clean_dataset
import shap

class PrivacyExplainabilityAnalysis:
    def __init__(self, original_pipeline, private_data, model):
        """
        Initialize with both original and private processed data
        Args:
            original_pipeline: AdultClassificationPipeline instance with processed data
            private_data: Processed private dataset
            model: Trained model
        """
        self.original_pipeline = original_pipeline
        self.private_data = private_data
        self.model = model
        
        # Initialize the SHAP explainer for tree-based models (e.g., RandomForest, XGBoost, etc.)
        self.explainer = shap.TreeExplainer(self.model)


    def get_natural_language_explanation(self, prompt):
        """
        Call the LLM API to get a natural language explanation based on the prompt.
        """
        url = "http://127.0.0.1:1234/v1/completions" # Corrected API endpoint
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "llama-3.2-1b-instruct",  # The model you specified
            "prompt": prompt,
  # The text you want to transform into human-readable explanation
            "max_tokens": 500,  # Limit to response length
            "temperature": 0.7  # Control creativity of response (0.7 is balanced)
        }

        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()  # Raise an error for bad status codes
            response_data = response.json()  # Get the response as a dictionary

            # Check if 'choices' key exists in the response
            if "choices" in response_data:
                return response_data["choices"][0]["text"].strip()  # Get the explanation text
            else:
                print(f"Unexpected response structure: {response_data}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error calling LLM API: {e}")
            return None
        except ValueError as e:
            print(f"Error parsing JSON response: {e}")
            return None

  

    def analyze_predictions(self, X_test, y_true):
        """
        Analyze model predictions and confidence levels
        """
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)
        confidence = np.max(y_prob, axis=1)
        
        wrong = y_pred != y_true
        high_conf_threshold = np.percentile(confidence[wrong], 75)
        wrong_but_confident = wrong & (confidence >= high_conf_threshold)
        
        return {
            'wrong': wrong,
            'wrong_but_confident': wrong_but_confident,
            'confidence': confidence,
            'threshold': high_conf_threshold,
            'predictions': y_pred,
            'probabilities': y_prob
        }

    def analyze_privacy_impact(self, wrong_but_confident_idx, X_test):
        """
        Analyze how privacy noise affects predictions
        """
        original_processed = self.original_pipeline.preprocess_features(self.original_pipeline.full_data)
        sensitive_features = ['age_binary', 'sex_Male']
        
        analysis = pd.DataFrame()
        for feature in sensitive_features:
            original = original_processed.loc[X_test.index][feature]
            noisy = X_test[feature]
            analysis[f'{feature}_changed'] = original != noisy
        
        return analysis.loc[wrong_but_confident_idx]

    def plot_confidence_distribution(self, results):
        """
        Plot confidence distribution for correct and incorrect predictions
        """
        plt.figure(figsize=(10, 6))
        plt.hist(results['confidence'][~results['wrong']], alpha=0.5, label='Correct', bins=20)
        plt.hist(results['confidence'][results['wrong']], alpha=0.5, label='Wrong', bins=20)
        plt.axvline(x=results['threshold'], color='r', linestyle='--', 
                   label=f'Threshold ({results["threshold"]:.2f})')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Distribution of Model Confidence')
        plt.legend()
        plt.savefig('assets/confidence_distribution.png')
        plt.close()

    def plot_feature_importance(self, feature_importance):
        """
        Plot feature importance visualization
        """
        plt.figure(figsize=(12, 6))
        importance_df = pd.DataFrame(feature_importance.items(), columns=['Feature', 'Importance'])
        importance_df = importance_df.sort_values('Importance', ascending=True).tail(10)
        
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Top 10 Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('assets/feature_importance.png')
        plt.close()

    def plot_feature_divergence(self, divergent_features):
        """
        Plot feature divergence in wrong predictions
        """
        plt.figure(figsize=(12, 6))
        divergence_df = pd.DataFrame(divergent_features.items(), columns=['Feature', 'Divergence'])
        divergence_df = divergence_df.sort_values('Divergence', ascending=True)
        
        sns.barplot(x='Divergence', y='Feature', data=divergence_df)
        plt.title('Feature Divergence in Wrong but Confident Predictions')
        plt.xlabel('Divergence Score')
        plt.tight_layout()
        plt.savefig('assets/feature_divergence.png')
        plt.close()

    def generate_natural_language_explanation(self, shap_values, top_n=5):
        """
        Generate a natural language explanation based on SHAP values, showing only the highest contributions.
        
        Args:
            shap_values: SHAP values for a given instance.
            top_n: Number of top features to display based on their contribution (default is 5).
        """
        # Get the features and their SHAP values
        feature_contributions = list(zip(shap_values.feature_names, shap_values.values))
        
        # Sort the contributions by absolute value, descending
        sorted_contributions = sorted(feature_contributions, key=lambda x: abs(x[1]), reverse=True)
        
        # Select top N features with the highest contributions
        top_contributions = sorted_contributions[:top_n]
        
        # Create the explanation text, including only the highest contributions
        explanation = f"Based on the model's analysis, the features that had the most impact on the prediction were as follows:\n"
        for feature, contribution in top_contributions:
            explanation += f"- {feature}: {contribution:.4f} contribution\n"
        
        # Prepare the prompt to pass to the LLM for natural language explanation
        prompt = (f"Explain the model's prediction for the chosen instance in simple terms, "
                f"highlighting the most important features and their contributions:\n{explanation}")
        
        # Call the LLM API to generate the natural language explanation
        natural_language_explanation = self.get_natural_language_explanation(prompt)
        
        return natural_language_explanation

    def analyze_and_explain(self, X_test, y_true, instance_idx):
        """
        Comprehensive analysis of model behavior and privacy impact
        """
        results = self.analyze_predictions(X_test, y_true)
        privacy_impact = self.analyze_privacy_impact(results['wrong_but_confident'], X_test)
        
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Compare feature distributions
        error_stats = X_test[results['wrong_but_confident']].describe()
        correct_stats = X_test[~results['wrong']].describe()
        
        # Calculate feature divergence
        feature_divergence = abs(error_stats.loc['mean'] - correct_stats.loc['mean'])
        divergent_features = feature_divergence.nlargest(5)
        
        # Generate an explanation for a specific instance using SHAP values
        shap_values = self.explainer(X_test.iloc[instance_idx:instance_idx+1])
        natural_language_explanation = self.generate_natural_language_explanation(shap_values[0])

        analysis_results = {
            'feature_importance': feature_importance.set_index('feature')['importance'].to_dict(),
            'divergent_features': divergent_features.to_dict(),
            'error_count': results['wrong'].sum(),
            'high_conf_errors': results['wrong_but_confident'].sum(),
            'privacy_changes': {
                'age': privacy_impact['age_binary_changed'].mean(),
                'sex': privacy_impact['sex_Male_changed'].mean()
            },
            'explanation': natural_language_explanation
        }
        
        self.plot_confidence_distribution(results)
        self.plot_feature_importance(analysis_results['feature_importance'])
        self.plot_feature_divergence(analysis_results['divergent_features'])
        return analysis_results

    def print_analysis(self, analysis_results):
        """
        Print analysis results in a clear format
        """
        print("\n=== Explainability Results ===")
        print(f"\nPrediction Statistics:")
        print(f"- Total errors: {analysis_results['error_count']}")
        print(f"- High confidence errors: {analysis_results['high_conf_errors']}")
        
        print("\nPrivacy Impact in Wrong but Highly Confident Instances:")
        print(f"- Age changed in {analysis_results['privacy_changes']['age']:.2%} of cases")
        print(f"- Sex changed in {analysis_results['privacy_changes']['sex']:.2%} of cases")
        
        print("\nTop Feature Importance:")
        for feature, importance in sorted(analysis_results['feature_importance'].items(), 
                                       key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {feature:<30} {importance:.3f}")
        
        print("\nMost Divergent Features in Wrong but Highly Confident Instances:")
        for feature, score in analysis_results['divergent_features'].items():
            print(f"  {feature:<30} {score:.3f}")
        
        print("\nNatural Language Explanation for Instance 0:")
        print(analysis_results['explanation'])

def main():
    # Load and prepare data
    data = read_dataset()
    clean_data = clean_dataset(data)
    
    # Create pipelines and process data
    pipeline = AdultClassificationPipeline(clean_data)
    processed_dataset, X_train, X_val, X_test, y_train, y_val, y_test = pipeline.prepare_data()
    
    private_pipeline = PrivateDatasetPipeline(processed_dataset)
    private_data = private_pipeline.create_private_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = private_pipeline.split_data(private_data)
    
    # Train model
    private_classification = AdultClassificationPipeline(private_data)
    private_classification.train_model(X_train, X_val, y_train, y_val)
    
    # Run analysis
    explainability = PrivacyExplainabilityAnalysis(pipeline, private_data, private_classification.model)
    
    # Provide the index of the instance you want to analyze, e.g., instance 0
    instance_idx = 0  # Example, you can choose a different index from X_test
    
    analysis_results = explainability.analyze_and_explain(X_test, y_test, instance_idx)
    explainability.print_analysis(analysis_results)

if __name__ == "__main__":
    main()
