# DT Probiotics Data Champion Assignment

## Overview

The DT Probiotics Data Champion app enables users to load and analyze datasets containing company information. It provides a comprehensive workflow for data cleaning, visualization, clustering, and prediction modeling, while also offering options to download the final outputs and the trained Random Forest model.This repository features a Streamlit application designed for analyzing company data and predicting prospects. It streamlines data cleaning, exploratory data analysis, clustering, and prediction modeling using Random Forest.


## Choose Your View

You can explore the project in one of two ways:

1. **Automated Dashboard with Streamlit [LINK](https://dt-probiotics-data-champion-assignment-7nedwf4hekxrj37nbpuappp.streamlit.app/)**: 
   - This is an interactive web application where you can explore the data and models, visualize various metrics, and download the processed data.
   - You will find models such as **Logistic Regression**, **Decision Trees**, **Random Forest**, and **Gradient Boosting** for prospect prediction.
   - The dashboard also features visualizations like scatter plots, correlation matrix heatmaps, and distributions of prospects.

2. **Google Colab Notebook [LINK](https://colab.research.google.com/github/Rahulaggl/DT-Probiotics-Data-Champion-Assignment/blob/main/dt_probiotics_data_champion_assignment.ipynb)**: 
   - This notebook allows you to run the code, experiment with different models, and explore the dataset step-by-step. 
   - It contains the data cleaning, feature engineering, model training, and evaluation steps, along with visualizations like the ones in the Streamlit app.
   - You can interact with the notebook directly in Google Colab, modify the code, and rerun the analysis as needed.

---

## Dataset

Source: Dataset [LINK](https://colab.research.google.com/github/Rahulaggl/DT-Probiotics-Data-Champion-Assignment/blob/main/DT_Probiotics_Data_Champion_Assignment.ipynb#scrollTo=eXK_LDHHBTzy)

Description: The dataset contains company information with the following columns:

Company Name: Name of the company.

Revenue: Annual revenue in USD.

Sector: The sector to which the company belongs.

Founded: Year the company was founded.

Other columns providing additional details about each company.

---


## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Dataset](#dataset)
6. [Technical Details](#technical-details)
7. [Insights](#insights)
8. [Contributing](#contributing)
9. [License](#license)



---

## Features

1. **Load Dataset**:
   - Import a dataset via a user-provided URL.
2. **Data Cleaning**:
   - Handle missing values and duplicates.
   - Transform data for analysis.
3. **Univariate and Bivariate Analysis**:
   - Generate distribution plots for numerical features.
   - Create correlation heatmaps.
4. **Feature Engineering**:
   - Add new features like revenue per year and revenue categories.
5. **Outlier Detection**:
   - Identify outliers using Z-scores.
6. **K-Means Clustering**:
   - Cluster data based on revenue and founding year.
   - Visualize clustering results.
7. **Prediction Modeling**:
   - Train a Random Forest model to predict revenue categories.
   - Evaluate model performance with metrics.
8. **Download Outputs**:
   - Export and download cleaned datasets and trained models.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Clone the Repository
```bash
https://github.com/Rahulaggl/DT-Probiotics-Data-Champion-Assignment.git
cd DT-Probiotics-Data-Champion-Assignment
```

### Install Required Libraries
```bash
pip install -r requirements.txt
```

---

## Usage

### Run the Application
```bash
streamlit run streamlit_app.py
```

### Interact with the Application
1. Enter the dataset URL in the sidebar.
2. Click **Load Dataset** to initiate the analysis.
3. Proceed with data cleaning, visualization, clustering, and modeling through the app.
4. Use the download buttons to save outputs and models.

---


## Technical Details

1. **Libraries Used**:
   - `streamlit`, `pandas`, `numpy`, `matplotlib`, `seaborn`
   - `scikit-learn` for clustering and prediction modeling
2. **Clustering**:
   - Features are scaled using StandardScaler.
   - K-Means is applied to group companies into three clusters.
3. **Prediction Modeling**:
   - A Random Forest Classifier predicts revenue categories.
   - Outputs include classification reports and confusion matrices.

---

## Insights

1. Companies with higher revenue per year typically have more established operations.
2. The Technology sector exhibits a higher concentration of companies with 'High' revenue.
3. Outliers in revenue, particularly on the higher end, were identified.
4. Clustering revealed distinct groups based on revenue and founding year.
5. The Random Forest model effectively categorized revenue classes based on company attributes.

---

## Contributing

Contributions are encouraged! Please open an issue or submit a pull request for enhancements or bug fixes.

---


## Conclusion

This project provides an end-to-end solution for analyzing and predicting company prospects using machine learning. You can explore the models and visualizations either through the interactive **Streamlit dashboard** or by running the code in **Google Colab**.


