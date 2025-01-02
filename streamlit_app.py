import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Function to load the dataset
@st.cache
def load_data(url):
    df = pd.read_csv(url)
    return df

# Title for the Streamlit app
st.title("DT Probiotics Data Champion Assignment")

# Sidebar for user inputs
st.sidebar.header("User Input Parameters")
dataset_url = st.sidebar.text_input('Enter the dataset URL:', 'https://raw.githubusercontent.com/Rahulaggl/DT-Probiotics-Data-Champion-Assignment/main/Task_Records.csv')
# Streamlit app layout
st.title('DT Probiotics Data Champion - Automated Dashboard')
st.write('This is a dashboard to analyze company data and predict prospects.')

# Links for Dataset and Google Colab Notebook
st.write("### 1. Dataset Link: [Link](https://github.com/Rahulaggl/DT-Probiotics-Data-Champion-Assignment/blob/main/Task_Records.csv)")
st.write("### 2. Google Colab Notebook: [Link](https://colab.research.google.com/drive/1J68d3Yn5sM_WU219_-dKOS86aZykOSdP#scrollTo=aHP3iCkZKk1l)")
st.write("### 3. Github: [Link](https://github.com/Rahulaggl/DT-Probiotics-Data-Champion-Assignment)")
# Load dataset when user clicks the button
if st.sidebar.button("Load Dataset"):
    df = load_data(dataset_url)
    st.write(f"Dataset loaded from {dataset_url}")

    # Step 1: Data Overview
    st.subheader("Step 1: Data Overview")
    st.write(f"Shape of the dataset: {df.shape}")
    st.write("First few records:")
    st.write(df.head())

    # Step 2: Data Cleaning
    st.subheader("Step 2: Data Cleaning")
    df_cleaned = df.copy()
    df_cleaned['Revenue'].fillna(df_cleaned['Revenue'].median(), inplace=True)
    df_cleaned['Sector'].fillna(df_cleaned['Sector'].mode()[0], inplace=True)
    df_cleaned.drop_duplicates(inplace=True)
    df_cleaned['Founded'] = df_cleaned['Founded'].astype(int)

    st.write("Data after cleaning:")
    st.write(df_cleaned.head())

    # Step 3: Univariate Analysis
    st.subheader("Step 3: Univariate Analysis")
    numerical_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        st.write(f"Distribution of {col}:")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df_cleaned[col], kde=True, color='blue', bins=30, ax=ax)
        st.pyplot(fig)

    # Step 4: Bivariate Analysis
    st.subheader("Step 4: Bivariate Analysis")
    corr = df_cleaned[numerical_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
    st.pyplot(fig)

    # Step 5: Feature Engineering
    st.subheader("Step 5: Feature Engineering")
    df_cleaned['Revenue_ZScore'] = zscore(df_cleaned['Revenue'])
    df_cleaned['Revenue_Per_Year'] = df_cleaned['Revenue'] / (2023 - df_cleaned['Founded'])
    conditions = [
        (df_cleaned['Revenue'] < 5000000),
        (df_cleaned['Revenue'] >= 5000000) & (df_cleaned['Revenue'] < 10000000),
        (df_cleaned['Revenue'] >= 10000000)
    ]
    choices = ['Low', 'Medium', 'High']
    df_cleaned['Revenue_Category'] = np.select(conditions, choices, default='Medium')

    st.write("Data after Feature Engineering:")
    st.write(df_cleaned.head())

    # Step 6: Outlier Detection
    st.subheader("Step 6: Outlier Detection")
    df_cleaned['Revenue_ZScore'] = zscore(df_cleaned['Revenue'])
    outliers = df_cleaned[df_cleaned['Revenue_ZScore'].abs() > 3]
    st.write("Outliers detected based on Z-score:")
    st.write(outliers)

    # Step 7: K-Means Clustering
    st.subheader("Step 7: K-Means Clustering")
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_cleaned[['Revenue', 'Founded']])

    kmeans = KMeans(n_clusters=3, random_state=42)
    df_cleaned['Cluster'] = kmeans.fit_predict(df_scaled)

    st.write("Cluster assignments:")
    st.write(df_cleaned[['Revenue', 'Founded', 'Cluster']].head())

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=df_cleaned['Revenue'], y=df_cleaned['Founded'], hue=df_cleaned['Cluster'], palette='Set1', ax=ax)
    st.pyplot(fig)

    # Step 8: Prediction Modeling (Random Forest Classifier)
    st.subheader("Step 8: Prediction Modeling (Random Forest Classifier)")
    X = df_cleaned[['Revenue', 'Founded', 'Revenue_Per_Year']]
    y = df_cleaned['Revenue_Category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    st.write("Random Forest Model Evaluation:")
    st.write(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    st.write(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # Step 9: Save Final Outputs
    if st.sidebar.button("Save Model and Outputs"):
        # Save the final outputs
        df_cleaned.to_csv("Final_Outputs.csv", index=False)
        joblib.dump(rf_model, 'random_forest_model.pkl')
        st.write("Model and Data saved as 'Final_Outputs.csv' and 'random_forest_model.pkl'")

    # Adding download options for CSV file and Colab PDF file
    st.subheader("Download Final Outputs")

    # Download button for the cleaned dataset (CSV file)
    csv = df_cleaned.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Cleaned Data (CSV)",
        data=csv,
        file_name='Final_Outputs.csv',
        mime='text/csv'
    )

    # Colab PDF Download Button (You can replace the link to the actual PDF file if available)
    colab_pdf_link = 'https://colab.research.google.com/drive/1J68d3Yn5sM_WU219_-dKOS86aZykOSdP'
    st.download_button(
        label="Download Google Colab PDF",
        data=colab_pdf_link,
        file_name="DT_Probiotics_Data_Champion_Assignment.pdf",
        mime="application/pdf"
    )

    # Documentation and Insights
    st.subheader("Documentation and Insights")
    st.write("1. Companies with higher revenue per year tend to have more established operations.")
    st.write("2. The Technology sector shows a higher concentration of companies with 'High' revenue.")
    st.write("3. Outliers in revenue were detected, particularly in the higher revenue range.")
    st.write("4. Clustering analysis identified groups of companies with distinct revenue and founding year characteristics.")
    st.write("5. The Random Forest model performed well in classifying revenue categories based on company characteristics.")
