# ❤️ Heart Disease Prediction using Machine Learning

> An end-to-end machine learning pipeline for predicting heart disease using the UCI Heart Disease dataset.
> This project demonstrates **data exploration, preprocessing, feature engineering, model training, evaluation, and experiment tracking**.

---

## 📌 Project Overview

Heart disease remains one of the leading causes of death worldwide. Early prediction can support faster clinical decisions and improve patient outcomes.

This project builds a complete **supervised machine learning classification pipeline** to predict whether a patient is likely to have heart disease based on clinical and diagnostic features.

### Objectives

* Understand and analyze a real-world healthcare dataset
* Perform **Exploratory Data Analysis (EDA)**
* Handle missing values and clean noisy data
* Apply **feature engineering** and preprocessing
* Train and compare multiple machine learning models
* Evaluate models using multiple performance metrics
* Save trained models and experiment outputs for reuse

---

## 🗂️ Dataset Information

**Dataset:** UCI Heart Disease Dataset
**Source:** Kaggle / UCI Machine Learning Repository
**Problem Type:** Binary Classification

### Sample Features

| Feature    | Description                           |
| ---------- | ------------------------------------- |
| `age`      | Age of patient                        |
| `sex`      | Gender                                |
| `cp`       | Chest pain type                       |
| `trestbps` | Resting blood pressure                |
| `chol`     | Serum cholesterol                     |
| `thalch`   | Maximum heart rate achieved           |
| `exang`    | Exercise induced angina               |
| `oldpeak`  | ST depression                         |
| `ca`       | Number of major vessels               |
| `thal`     | Thalassemia status                    |
| `num`      | Target label (heart disease presence) |

---

# 🔍 Exploratory Data Analysis (EDA)

EDA was performed to understand the dataset before model development.

### Completed analyses

✅ Dataset shape and structure inspection
✅ Data type verification
✅ Missing value analysis
✅ Descriptive statistics (mean, std, min, max)
✅ Distribution analysis using histograms
✅ Outlier detection using boxplots
✅ Correlation analysis and heatmap
✅ Class imbalance checking

### Key insights

* Significant missing values were found in several clinical columns
* Numerical feature distributions were not fully uniform
* Some outliers were present in cholesterol and blood pressure
* Several features showed useful correlation with the target label

---

# 🧹 Data Preprocessing

Proper preprocessing was essential before training models.

### Steps performed

#### 1. Missing Value Handling

* Initial row deletion caused major data loss
* Final approach used:

  * **Median imputation** for numerical features
  * **Mode imputation** for categorical features

#### 2. Categorical Encoding

Applied encoding to convert non-numeric features into machine-readable format.

Techniques used:

* Label Encoding
* One-Hot Encoding

#### 3. Feature Scaling

Normalization/standardization is applied to numerical features to ensure balanced model learning.

Used for models sensitive to feature magnitude, such as:

* Logistic Regression
* Decision Tree
* Random Forest
* KNN
* SVM
* XGBoost

---

# ⚙️ Feature Engineering

Additional feature transformations were explored to improve predictive power.

Examples:

* High cholesterol indicator
* High blood pressure indicator
* Age-based risk grouping
* Combined clinical risk signals

Feature engineering helped expose patterns not immediately visible in raw data.

---

# 🤖 Machine Learning Models

Multiple models were trained and compared.

## Implemented Models

### 1. Logistic Regression

* Baseline interpretable model
* Good for binary classification

### 2. Decision Tree

* Rule-based learning
* Easy to visualize and explain

### 3. Random Forest

* Ensemble learning method
* Improved stability and accuracy

### 4. K-Nearest Neighbors (KNN)

* Distance-based classification
* Highly dependent on scaling

### 5. Support Vector Machine (SVM)

* Strong classification boundary learner
* Effective on structured datasets

### 6. XGBoost *(experimental / optional)*

* Gradient boosting model
* Advanced high-performance classifier

---

# ✂️ Train-Test Split

Dataset split strategy:

* **80% Training Data**
* **20% Testing Data**

### Why?

To evaluate model performance on unseen data and reduce overfitting risk.

---

# 📊 Model Evaluation

Multiple metrics were used for robust assessment.

### Performance Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Cross Validation Score

### Why multiple metrics?

For medical diagnosis, **Recall** is especially important because failing to detect an actual disease case can be costly.

---

# 📈 Visualizations

This project includes multiple visual outputs:

* Histograms
* Boxplots
* Correlation Heatmaps
* Confusion Matrix Heatmaps
* Model Performance Comparison Charts

These visualizations improve interpretability and help compare model behavior.

---

# 🐞 Challenges Faced

### Missing Value Problem

Initial row deletion reduced the dataset size significantly.

**Fix:** Imputation methods replaced missing values.

---

### Categorical Data Handling

Machine learning models could not directly process text values.

**Fix:** Applied encoding techniques.

---

### Pipeline Order Errors

Errors such as `X_train not defined` occurred during development.

**Fix:** Reorganized pipeline execution order.

---

### JSON Serialization Issues

Evaluation outputs were not immediately JSON-compatible.

**Fix:** Converted NumPy objects to native Python types.

---

# 💾 Saved Outputs

Project artifacts include:

* Trained model files
* Prediction outputs
* Evaluation metrics (JSON)
* Error logs
* Experiment results

---

# 🧠 Key Learnings

Through this project, I gained practical experience in:

* Real-world dataset analysis
* Missing data handling
* Feature preprocessing
* Feature engineering
* Model comparison
* Performance evaluation
* Experiment reproducibility
* Debugging ML pipelines

---

# 🚀 Future Improvements

Potential future enhancements:

* Full hyperparameter tuning (GridSearchCV)
* Feature selection techniques
* Model explainability (SHAP / feature importance)
* Web deployment for live prediction
* Additional healthcare datasets for validation

---

# 🛠️ Tech Stack

```text
Python
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
XGBoost
JSON
Jupyter Notebook
```

---

# 📂 Project Structure

```text
heart-disease-ml/
├── data/
├── notebooks/
├── models/
├── outputs/
├── metrics/
├── logs/
├── README.md
└── requirements.txt
```

---

# 👨‍💻 Author

**Ashfaq Tannbir**
Electrical & Electronic Engineering graduate
Exploring Machine Learning, Data Science, and Intelligent Systems

---

# ⭐ Repository Purpose

This repository is designed not only to build a predictive model, but also to serve as a **learning resource** for beginners who want to understand how an end-to-end machine learning project is developed—from raw data to final evaluation.

If this project helps you learn something, feel free to explore, fork, and improve it.
