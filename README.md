Data Preprocessing and Machine Learning Model
Overview
This Python project involves data preprocessing and building a machine learning model to classify a dataset into categories based on various features. The script includes steps for data loading, cleaning, visualization, outlier removal, feature selection, and model training, followed by performance evaluation and the ability to download the trained model.

Features
Data Loading: Loads a CSV file either from Google Colab or your local machine.
Exploratory Data Analysis (EDA): Generates descriptive statistics, histograms, boxplots, and correlation heatmaps for initial analysis.
Data Preprocessing: Handles missing values, removes outliers, and performs feature selection.
Data Normalization: Standardizes features for better model performance.
SMOTE: Balances the dataset using Synthetic Minority Over-sampling Technique (SMOTE).
Modeling: Trains a HistGradientBoostingClassifier and evaluates its performance.
Visualization: Plots actual vs. predicted values for model evaluation.
Model Saving: Saves the trained model and allows the user to download it.
Requirements
Before running this code, ensure that the following libraries are installed:

bash
Αντιγραφή κώδικα
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn joblib
How to Use
Step 1: Load the Dataset
The script will prompt you to upload your CSV file. Depending on your environment (Google Colab or local Python setup), the method to load the file will differ:

In Google Colab: You'll be prompted to upload the CSV file directly through the browser.
In local Python environments (e.g., VSCode): You'll be asked to input the file path.
Step 2: Explore the Data
The script will perform exploratory data analysis (EDA) and display:

Descriptive statistics of the dataset.
Histograms of all features.
Boxplots for outlier detection.
A correlation heatmap for numeric features.
Step 3: Data Preprocessing
Missing Data: Missing values will be imputed with the mean of each column.
Outlier Removal: Outliers will be removed based on the 99th percentile of each numeric column.
Feature Selection: Selected features for training will include columns like age, chol, cp, etc., and the target variable target.
Step 4: Data Balancing (SMOTE)
The script applies SMOTE to balance the dataset by generating synthetic samples for the minority class.

Step 5: Train the Model
The script uses a HistGradientBoostingClassifier model for classification. After training, it evaluates the model on the test set using:

Accuracy Score
Classification Report (Precision, Recall, F1-Score)
ROC-AUC Score
Step 6: Visualize Results
The script will generate a plot comparing actual vs. predicted values to evaluate model performance.

Step 7: Save and Download the Model
In Google Colab: The trained model will be saved and automatically downloaded as a .pkl file.
In local Python environments (VSCode): You'll be prompted to select a location to save the model as a .pkl file.
Example Output
bash
Αντιγραφή κώδικα
The file was loaded successfully!
First 5 rows:
   age  trestbps  chol  ...  slope  target
0   63       145   233  ...      2       1
1   37       130   250  ...      2       1
2   41       130   204  ...      1       1
3   56       120   236  ...      1       0
4   57       120   354  ...      1       1

Accuracy on Test Set: 0.8521

Classification Report:
              precision    recall  f1-score   support
           0       0.86      0.84      0.85        40
           1       0.84      0.86      0.85        38

ROC AUC: 0.9045
Notes
Ensure the dataset has a target column (e.g., target) for classification tasks.
The script uses a HistGradientBoostingClassifier model, but you can experiment with other models as needed.
The dataset must include numeric features and a target variable.
