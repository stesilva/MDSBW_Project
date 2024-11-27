# Adult Income Classification Pipeline

This repository contains the implementation of the first and second part of the Adult Income Classification project, which focuses on building a fair and privite classification pipeline for predicting income levels.

## Project Structure

```
.
├── adult.csv           # Dataset file
├── classifier.py       # Main implementation file
└── README.md          # This file
```

## Part 1: Basic Classification Implementation
### Features
1. **Data Loading and Preprocessing**
   - Load Adult dataset with appropriate data types
   - Handle missing values
   - Remove unnecessary features (fnlwgt)
   - Standardize column names

2. **Feature Processing**
   - Age binarization (as required by project)
   - Numerical feature standardization
   - Categorical feature encoding using one-hot encoding
   
3. **Data Splitting**
   - Train/Validation/Test split (60%/20%/20%)
   - Stratified sampling to maintain class distribution

4. **Model Training**
   - LightGBM classifier implementation
   - Early stopping with validation set
   - Basic hyperparameter configuration

5. **Model Evaluation**
   - Multiple evaluation metrics (Accuracy, Precision, Recall, F1-score)
   - Classification report for detailed performance analysis
   - Feature importance visualization

## Part 2: Fairness Analysis
### Features
1. **Original Model Fairness Analyses**
   - Verifies two fairness metrics based on the test dataset and the predicted label
   - Report findings of the two metrics

2. **Fairness Technique to Ensure Fair Classifier**
   - Adversarial Debiasing (make the model less reliant on sensitive features)
   - Main Network (Predictor): A standard model (e.g., logistic regression, neural network)
   - Adversarial Network: A separate network is introduced to predict the sensitive feature (sex)

3. **New Model Evaluation**
   - Multiple evaluation metrics (Accuracy, Precision, Recall, F1-score)
   - Classification report for detailed performance analysis

4. **New Model Fairness Analyses**
   - Verifies two fairness metrics based on the test dataset and the new predicted label
   - Report findings of the two metrics
     
## Part 3: Privacy Analysis
### Features
1. **Assessing the Current State of Sensitive Attributes**
   - Extract the preprocessed data on attributes Age and Sex
   - Compute the cross-tabulation

2. **Applying Local Differential Privacy Technique**
   - Apply local differential privacy using randomised response
   - Select the most appropriate values for truth probabilities
   - Create a private dataset

3. **Dataset Comparison**
   - Compute cross-tabulation for private dataset
   - Calculate the absolute and relative errors in comparison with the original dataset

4. **New Model Evaluation**
   - Split the private dataset as in Part 1
   - Train and evaluate the same model with private data
   - Report findings   

### Usage
```python
# Run the classification pipeline
python classifier.py
```

## Current Results and Limitations
## Part 1: Basic Classification Implementation
### Performance Metrics
- Accuracy: ~0.83
- Significant disparity between classes:
  - Class 0 (≤50K): Precision: 0.95, Recall: 0.82
  - Class 1 (>50K): Precision: 0.61, Recall: 0.86

### Identified Issues
1. **Class Imbalance**
   - Imbalanced class distribution affecting model performance
   - Higher performance on majority class (≤50K)
   - Lower precision for minority class (>50K)

2. **Protected Attributes**
   - Potential bias in predictions regarding age and gender
   - Need for fairness analysis in subsequent parts

3. **Privacy Concerns**
   - Sensitive attributes (age, gender) require privacy protection
   - Current implementation doesn't address privacy requirements

## Part 2: Fair Classification Implementation
### Performance Metrics
- Accuracy: ~0.84
- Significant disparity between classes:
  - Class 0 (≤50K): Precision: 0.85, Recall: 0.96
  - Class 1 (>50K): 0.78, Recall: 0.47

### Fairness Metrics
   - Disparate Impact (Focuses on the relative rate of positive outcomes between groups, aiming for a ratio close to 1 to indicate fairness): 
      - Before Adversarial Debiasing: 0.30 (indicating significant bias against the protected group)
      - After Adversarial Debiasing: 0.81 (showing a much more balanced result, closer to fairness)
   - Statistical Parity Difference (Focuses on the absolute difference between the groups’ probabilities of receiving a positive outcome, aiming for a value close to 0 to indicate fairness): 
      - Before Adversarial Debiasing: -0.31 (suggesting a strong imbalance in outcomes between the groups)
      - After Adversarial Debiasing: -0.03 (a much smaller disparity, indicating a more fair model)

### Identified Issues
1. **Class Imbalance**
   - Strong performance on Class 0, but poor recall for Class 1 (>50K)
   - New model fails to predict the minority class effectively

2. **Protected Attributes**
   - Tried using both protected attributes to mitigate unfairness in the model, but the best performance was using only one attribute in the new model (sex)


## Part 3: Privacy Classification Implementation
### Performance Metrics
- Accuracy: ~0.82
- Significant disparity between classes holds:
  - Class 0 (≤50K): Precision: 0.95, Recall: 0.81
  - Class 1 (>50K): 0.59, Recall: 0.87

### Local Differential Privacy Metrics
   - Randomised Response on Age:
      - Since the data is equally distributed, the values of p and q were chosen to be equal. Namely, p=q=0.95
   - Randomised Response on Sex:
      - Since the data is skewed towards male population (proporion of females ~34%), q (probability of reporting sex=Female) was chosen to be lower than p to introduce higher privacy to the less common data.
   - Overall, pretty high values of p and q were introduced (with epsilon ~3) in both attributes in order not to deviate from the original proportions of data. This was the result of analysing the cross-tabulation of original and private datasets and relative errors.

### Model Performance
- As a result, the accuracy was not much affected (0.82 compared to 0.83 in the original classifier).
- Although, it should be noted that even with the introduction of lower probabilities p and q, similar results were obtained, i.e. low impact on model performance. This could be explained by age and sex attributes being not of that high importance as features.

## Dependencies
- Python 3.x
- pandas
- numpy
- scikit-learn
- LightGBM
- matplotlib
- aif360

## Installation
```bash
pip install pandas numpy scikit-learn lightgbm matplotlib aif360
```

## Notes
- Current implementation focuses on basic classification pipeline and fair analysis
- Subsequent parts will address fairness, privacy, and explainability
- Model performance serves as baseline for future improvements

## Citation
Adult Income dataset from UCI Machine Learning Repository:
https://archive.ics.uci.edu/ml/datasets/adult
