# Used Car Price Prediction

This project implements a Machine Learning pipeline to predict used car prices using a **Random Forest Regressor**. The model is optimized through systematic hyperparameter tuning.

## 🚀 Project Pipeline

### 1. Exploratory Data Analysis (EDA)
* **Data Loading**: Using `pandas` to handle the "CarPrice_Assignment.csv" dataset.
* **Feature Identification**: Categorizing variables into Numerical (e.g., horsepower, price) and Categorical (e.g., carbody, fueltype).
* **Visualization**: Distribution analysis using histograms to identify skewness and potential outliers in engine size, price, and horsepower.

### 2. Data Pre-processing
* **Categorical Encoding**: Applied **One-Hot Encoding** to transform categorical labels into a machine-readable binary format.
* **Feature Selection**: Calculated a correlation matrix to select features with a high impact on price (threshold $|r| > 0.6$). Selected features include `enginesize`, `curbweight`, `horsepower`, and `carwidth`.

### 3. Model Training & Tuning
* **Algorithm**: Random Forest Regressor.
* **Hyperparameter Tuning**: Used `GridSearchCV` with 5-fold cross-validation to find the optimal configuration:
  * `max_depth`: 10
  * `n_estimators`: 150
  * `min_samples_leaf`: 2

### 4. Validation & Metrics
The model was evaluated on a 20% test set with the following results:
* **R-squared (R²)**: ~0.96 (The model explains 96% of the price variance).
* **Cross-Validation**: Robustness check to evaluate generalization across different data subsets.

## 🛠️ User Interface
The notebook includes an **Interactive Prediction Tool**. Users can input car specifications in **Metric Units** (cc, kg, cm, CV), which are automatically converted into the model's native units for real-time price estimation.

## 📦 Requirements
* Python 3.x
* Scikit-Learn
* Pandas
* Matplotlib / Seaborn
