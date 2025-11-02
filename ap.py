import io
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Student Performance Prediction", layout="wide")

st.title("Student Performance Prediction")

uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    try:
        content = uploaded_file.getvalue().replace(b'\x00', b'')
        df = pd.read_csv(
            io.BytesIO(content),
            sep=None,
            engine='python',
            encoding='latin1',
            on_bad_lines='skip'
        )

        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        st.write("Shape:", df.shape)
        st.write("Columns:", list(df.columns))

        le = LabelEncoder()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = le.fit_transform(df[col])

        if 'G3' not in df.columns:
            st.error("Column 'G3' not found in dataset.")
        else:
            df['Pass'] = np.where(df['G3'] >= 10, 1, 0)

            X = df.drop(['G3', 'Pass'], axis=1)
            y = df['Pass']

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            dt = DecisionTreeClassifier(random_state=42)
            dt.fit(X_train, y_train)
            y_pred_dt = dt.predict(X_test)
            acc_dt = accuracy_score(y_test, y_pred_dt)
            mse_dt = mean_squared_error(y_test, y_pred_dt)
            rmse_dt = np.sqrt(mse_dt)
            r2_dt = r2_score(y_test, y_pred_dt)

            nb = GaussianNB()
            nb.fit(X_train, y_train)
            y_pred_nb = nb.predict(X_test)
            acc_nb = accuracy_score(y_test, y_pred_nb)
            mse_nb = mean_squared_error(y_test, y_pred_nb)
            rmse_nb = np.sqrt(mse_nb)
            r2_nb = r2_score(y_test, y_pred_nb)

            results = pd.DataFrame({
                'Model': ['Decision Tree', 'Naive Bayes'],
                'Accuracy': [acc_dt, acc_nb],
                'MSE': [mse_dt, mse_nb],
                'RMSE': [rmse_dt, rmse_nb],
                'R2 Score': [r2_dt, r2_nb]
            })

            st.subheader("Model Performance Comparison")
            st.dataframe(results.style.format({"Accuracy": "{:.3f}", "MSE": "{:.3f}", "RMSE": "{:.3f}", "R2 Score": "{:.3f}"}))

            fig, ax = plt.subplots(figsize=(4, 2.5))
            sns.barplot(x='Model', y='Accuracy', data=results, palette='Blues_d', ax=ax)
            ax.set_title("Model Accuracy Comparison", fontsize=10)
            ax.set_ylim(0, 1)
            st.pyplot(fig)

            st.subheader("Make a Prediction")

            input_data = {}
            for col in X.columns:
                if df[col].nunique() < 10:
                    val = st.selectbox(f"{col}", sorted(df[col].unique()))
                else:
                    val = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
                input_data[col] = val

            if st.button("Predict"):
                input_df = pd.DataFrame([input_data])
                input_scaled = scaler.transform(input_df)
                pred_dt = dt.predict(input_scaled)[0]
                pred_nb = nb.predict(input_scaled)[0]

                st.write("Decision Tree Prediction:", "Pass" if pred_dt else "Fail")
                st.write("Naive Bayes Prediction:", "Pass" if pred_nb else "Fail")

    except Exception as e:
        st.error(f"Could not load the file: {e}")

else:
    st.info("Please upload a dataset to continue.")
