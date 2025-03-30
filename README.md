# 📊 Alphabet Soup Charity Funding Prediction

## 🧠 Overview of the Analysis

The nonprofit foundation **Alphabet Soup** seeks to identify organizations that are most likely to succeed after receiving funding. The goal of this project is to build a binary classification model using deep learning that predicts whether an applicant organization will be successful based on a variety of features.

Using a dataset of over 34,000 historical funding records, we cleaned and prepared the data, built a neural network model using TensorFlow and Keras, optimized it through multiple strategies, and evaluated its performance. Our objective was to achieve a prediction accuracy of **at least 75%**.

---

## 📈 Results

### 🔹 Data Preprocessing

- **Target variable**: `IS_SUCCESSFUL`  
- **Feature variables**: All other columns, except those removed below
- **Removed columns**:  
  - `EIN` and `NAME` — identification only  
  - `ASK_AMT`, `STATUS`, `SPECIAL_CONSIDERATIONS`, `INCOME_AMT` — removed during optimization for low predictive power or noise

- **Encoding**:  
  All categorical variables were converted using `pd.get_dummies()`

- **Split**:  
  Data was split into training and test sets, and scaled with `StandardScaler`.

---

### 🔹 Neural Network Architecture

- **Initial model (non-optimized)**:
  - Hidden layers: 2
  - Neurons: 80 → 30
  - Activations: ReLU (hidden), Sigmoid (output)
  - Performance: ~65.3% accuracy

- **Optimized model**:
  - Hidden layers: 3
  - Neurons: 128 → 64 → 32
  - Activations: LeakyReLU (hidden), Sigmoid (output)
  - Regularization: Dropout layers (0.2) after first two hidden layers
  - Training: 150 epochs
  - Performance: **66.3% accuracy**

---

### 🔹 Optimization Attempts

We made at least three distinct optimization attempts:

1. **Architecture Tuning**
   - Added a third hidden layer
   - Increased number of neurons per layer
2. **Regularization**
   - Introduced `Dropout` layers to prevent overfitting
   - Used `LeakyReLU` activations instead of standard ReLU
3. **Feature Selection**
   - Dropped noisy and low-impact columns like `ASK_AMT`, `STATUS`, and `INCOME_AMT`

Despite these efforts, the model plateaued around **66.3% accuracy**, below the target.

---

## ✅ Final Evaluation

- **Final accuracy**: ~66.3%
- **Final loss**: ~3.33
- **File saved as**: `AlphabetSoupCharity_Optimization.h5`

---

## 💡 Recommendation

Based on our results, a deep learning model did not achieve the desired performance. For this tabular data structure, we recommend trying a **tree-based ensemble method**, such as:

- **Random Forest**
- **XGBoost**

These models are often more effective for structured datasets, are easier to tune, and provide better interpretability through feature importance metrics.

---

## 📁 Files

- `Starter_Code.ipynb` – Initial preprocessing and modeling
- `Starter_Code_optimized.ipynb` – Optimized neural network version
- `AlphabetSoupCharity.h5` – Initial model file
- `AlphabetSoupCharity_Optimization.h5` – Final optimized model
