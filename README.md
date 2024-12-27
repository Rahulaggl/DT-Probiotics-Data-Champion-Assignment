# DT Probiotics Data Champion Assignment

## Overview

This repository contains a data analysis and machine learning project focused on predicting company prospects based on various features like revenue, growth rate, and industry. The assignment includes an interactive **Streamlit Dashboard** for real-time analysis, and a **Google Colab** notebook for running the code and performing the data analysis.

## Choose Your View

You can explore the project in one of two ways:

1. **Automated Dashboard with Streamlit [LINK](https://dt-probiotics-data-champion-assignment-7nedwf4hekxrj37nbpuappp.streamlit.app/)**: 
   - This is an interactive web application where you can explore the data and models, visualize various metrics, and download the processed data.
   - You will find models such as **Logistic Regression**, **Decision Trees**, **Random Forest**, and **Gradient Boosting** for prospect prediction.
   - The dashboard also features visualizations like scatter plots, correlation matrix heatmaps, and distributions of prospects.

2. **Google Colab Notebook [LINK](https://colab.research.google.com/drive/1J68d3Yn5sM_WU219_-dKOS86aZykOSdP#scrollTo=aHP3iCkZKk1l)**: 
   - This notebook allows you to run the code, experiment with different models, and explore the dataset step-by-step. 
   - It contains the data cleaning, feature engineering, model training, and evaluation steps, along with visualizations like the ones in the Streamlit app.
   - You can interact with the notebook directly in Google Colab, modify the code, and rerun the analysis as needed.

---

## Dataset

The dataset used in this project is available in the following [GitHub link](https://github.com/Rahulaggl/DT-Probiotics-Data-Champion-Assignment/blob/main/Task_Records.csv). It contains several company records, with features like:

- **Company_ID**
- **Company_Name**
- **Revenue**
- **Growth_Rate**
- **Industry**
- **Location**
- **Prospect** (Target variable for prediction)

Before using the dataset, ensure you have cleaned and processed it (as done in the project) to remove duplicates and handle missing values.

---

## Project Structure

- **Streamlit Dashboard**: A live web app to interact with the data and models.
- **Google Colab Notebook**: A Jupyter-like notebook to run and modify the code.
- **Dataset**: A CSV file containing company data used for prediction.
- **ML Models**: Logistic Regression, Decision Trees, Random Forest, and Gradient Boosting.

---

## Running the Project (Streamlit App)

The **Streamlit dashboard** is hosted on Streamlit Share, and you don't need to install any dependencies to run it locally.

1. Visit the live app at: [Streamlit App](https://dt-probiotics-data-champion-assignment-7nedwf4hekxrj37nbpuappp.streamlit.app/).
2. Interact with the models, visualizations, and explore the processed data.
3. You can download the processed data by clicking the download button within the app.

---

## Running the Project in Google Colab

If you'd like to run the code in **Google Colab**, follow these steps:

1. Open the notebook using [this link](https://colab.research.google.com/drive/1J68d3Yn5sM_WU219_-dKOS86aZykOSdP#scrollTo=aHP3iCkZKk1l).
2. Mount Google Drive (if needed) to save or load files.
3. Install the required libraries:
   ```bash
   !pip install pandas numpy seaborn matplotlib scikit-learn
   ```
4. Run the cells in the notebook step by step, ensuring all sections (data loading, preprocessing, model training, and evaluation) are completed.

---

## Key Features of the Project

1. **Data Cleaning and Preprocessing**: 
   - Handling missing values, removing duplicates, and encoding categorical variables.
   - The "Prospect" column is derived based on the Growth Rate (e.g., companies with growth rate > 20% are classified as 'Yes').

2. **Machine Learning Models**: 
   - Logistic Regression, Decision Trees, Random Forest, and Gradient Boosting are used to predict company prospects.
   - Metrics like **accuracy**, **confusion matrix**, and **classification report** are used to evaluate the models.

3. **Visualizations**:
   - **Scatter plot** showing the relationship between Revenue and Growth Rate.
   - **Correlation matrix heatmap** to understand the relationships between numerical features.
   - **Prospect distribution** by **Industry** and **Location**.
   - **Revenue and Growth Rate** distribution for a better understanding of the data.

4. **Download Processed Data**: 
   - The processed dataset can be downloaded for further analysis.

---

## Requirements

- **Python 3.x** (preferably Python 3.10 or higher)
- **Required Libraries**:
  - pandas
  - numpy
  - seaborn
  - matplotlib
  - scikit-learn

You don't need to install any additional libraries to run the **Streamlit App** as it is already hosted online. However, if you want to work locally with **Google Colab**, you can install the required libraries.

---

## Conclusion

This project provides an end-to-end solution for analyzing and predicting company prospects using machine learning. You can explore the models and visualizations either through the interactive **Streamlit dashboard** or by running the code in **Google Colab**.


