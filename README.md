```markdown
# 🚖 Uber Fare Price Prediction using Deep Learning (PyTorch & TensorFlow)

## 📌 Project Overview

This project aims to build a neural network model that predicts **Uber ride fare prices** using trip-related features such as:

- Pickup & drop-off coordinates
- Distance between locations
- Passenger count
- Time features (hour, weekday, month, year)

The model learns patterns from historical ride data to estimate accurate fare prices for new trips using **PyTorch** (and experiments with TensorFlow concepts).

---

## 🎯 Problem Statement

Accurately predicting ride fares is important for dynamic pricing systems, customer transparency, and business optimization.  
The goal is to train a deep learning model that can generalize fare estimation from structured trip data.

---

## 📊 Dataset Description

The dataset contains **159,990 samples** with **51 features**, including:

- Geographical features: pickup and drop-off coordinates
- Engineered features: distance, year, hour, weekday, month encoding
- Target variable: fare_amount

### Key Properties:
- No missing values
- Fully numeric dataset
- Standardized features
- High-dimensional time encoding

---

## 🔎 Exploratory Data Analysis (EDA)

The following analyses were performed:

- Data shape and structure inspection
- Feature type analysis
- Missing value check (none found)
- Statistical summary using `describe()`
- Boxplots for outlier detection
- Correlation heatmap of features
- Target variable distribution analysis

### Observations:
- Fare distribution contains extreme outliers
- Some values required filtering (negative and very high fares)
- Feature scaling is essential for neural networks

---

## 🧹 Data Preprocessing

### Steps Applied:

#### 1. Data Cleaning
- Removed invalid fare values (`fare_amount <= 0`)
- Removed extreme outliers (`fare_amount > 23`)

#### 2. Feature Scaling
- Applied `StandardScaler` to normalize features

#### 3. Dimensionality Reduction
- Applied **PCA (95% variance retention)** to reduce feature space

### Final Pipeline:
StandardScaler → PCA (95% variance)

---

## 🧠 Model Architecture

A fully connected deep neural network built with PyTorch:

```

Input (PCA features ~50)
→ Linear(50 → 128) + ReLU
→ Linear(128 → 64) + ReLU
→ Linear(64 → 32) + ReLU
→ Linear(32 → 16) + ReLU
→ Linear(16 → 1)

```

---

## ⚙️ Training Configuration

- Framework: PyTorch
- Optimizer: Adam
- Learning Rate: 0.00001
- Batch Size: 50
- Loss Function: L1 Loss (MAE)
- Device: CUDA (GPU)

---

## 📈 Validation Strategy

Validation is performed using:

- `torch.no_grad()` for inference
- L1 Loss (Mean Absolute Error)
- Average loss across batches

---

## 📦 Project Pipeline

```

Raw Data
↓
EDA & Cleaning
↓
StandardScaler
↓
PCA (95% variance)
↓
Neural Network Training
↓
Validation Evaluation
↓
Prediction

````

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn
````

---

### 2. Load Data

Make sure your dataset files are in the working directory:

* train.csv
* val.csv
* test.csv

---

### 3. Train Model

```bash
python train.py
```

---

### 4. Evaluate Model

```bash
python evaluate.py
```

---

## 📊 Key Learnings

* Feature scaling is critical for deep learning on tabular data
* PCA reduces noise and improves training stability
* Outlier handling significantly improves regression performance
* Neural networks can effectively model structured pricing problems

---

## 🔮 Future Improvements

* Replace PCA with learned embeddings
* Try Gradient Boosting models (XGBoost / LightGBM)
* Hyperparameter tuning (learning rate, layers, dropout)
* Deploy model using FastAPI or Flask
* Build real-time fare prediction API
* Dockerize the full pipeline

---

## 👨‍💻 Author

**Tewodros Bewuket**
MSc AI Student | Backend Engineer | AI Engineer
```
