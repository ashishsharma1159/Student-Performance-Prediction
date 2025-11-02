Student Achievement Prediction using Decision Tree and Naive Bayes

This particular project is an implementation intended to predict whether a student will pass or fail school based on academic level and student/family characteristics. Identifying students to be at risk of failing school allows teachers and administrators to develop interventions and acknowledge the specific capabilities and support they will require to succeed in school, if any. There are two different classification methods being evaluated in this project; Decision Tree and Naive Bayes.

An interactive Streamlit web application has been developed for making predictions using uploaded data.

This classroom project uses a dataset from the UCI Machine Learning Repository titled Student Performance Data Set. The dataset contains 649 records and has 33 attributes (demographic, academic, and family background to name a few). Before the project used the data, however, null values and malformed lines were removed from the dataset. All categorical variables were encoded using a method called LabelEncoder. We created a binary target column called Pass, where 1 = if G3 (the final grade), was greater than or equal to 10 and 0 if it was less than 10. All features were standardized with the StandardScaler method. A training set (80%) and a test set (20%) split was created from the data provided.

DATASET SOURCE:[https://archive.ics.uci.edu/ml/datasets/Student+Performance](https://archive.ics.uci.edu/ml/datasets/Student+Performance)

PREPROCESSING STEPS:
Removed null and malformed lines (if any)
Encoded categorical variables using LabelEncoder
Created a new binary target column:
  Pass = 1 if G3 ≥ 10
  Pass = 0 if G3 < 10
Standardized numerical features using StandardScaler
Split data into training and test sets (80%–20%)

MODELS:
1)Decision Tree Classifier
A non-parametric supervised learning method used for classification.
Advantages: Handles nonlinear relationships and feature interactions.

2)Naive Bayes Classifier
A probabilistic model based on Bayes’ theorem assuming feature independence.
Advantages: Fast, simple, and effective with small datasets.

 Model Workflow
1. Load and clean dataset  
2. Encode categorical and scale numerical data  
3. Split into training and test sets  
4. Train Decision Tree and Naive Bayes models  
5. Evaluate and compare performance (Accuracy, MSE, RMSE, R²)

Steps to Run the Code
1. Clone the repository
git clone https://github.com/ASHISHSHARMA1159/Student-Performance-Prediction.git
cd student-performance-prediction

2.CREATE AND ACTIVATE YOUR VIRTUAL ENVIRONMENT
python -m venv env
source env/bin/activate #FOR MAC
env\Scripts\activate    #FOR WINDOWS

3.INSTALL ALL DEPENDANCIES
pip install -r requirements.txt

4.RUN THE STREAMLIT APP
streamlit run student_performance.py

5.UPLOAD YOUR DATASET
download dataset student-mat.csv or student-por.csv and upload on the app

Experiments and Results
Model	Accuracy	MSE	RMSE	R² Score
Decision Tree	0.89	0.11	0.33	0.58
Naive Bayes	0.84	0.16	0.40	0.45
(Values may vary slightly depending on dataset split.)

Conclusion
This project demonstrates that both Decision Tree and Naive Bayes are effective in predicting student outcomes.
However, the Decision Tree Classifier achieved superior performance in terms of accuracy and error metrics.
Key Learnings:
Data preprocessing (especially encoding and scaling) significantly impacts results.
Simple models like Naive Bayes can still perform well in educational prediction tasks.
Visualization and interpretability are crucial when presenting model results to educators.

REFERENCES
Cortez, P., & Silva, A. M. G. (2008). Using Data Mining to Predict Secondary School Student Performance.
UCI Machine Learning Repository
Scikit-learn Documentation — https://scikit-learn.org
Streamlit Documentation — https://docs.streamlit.io
