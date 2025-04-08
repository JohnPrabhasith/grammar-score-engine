# 🎧 Grammar Scoring Engine from Audio Samples

This project implements a **Grammar Scoring Engine** that predicts grammar proficiency scores from audio samples using traditional machine learning models (Random Forest, XGBoost, LightGBM). Audio files are transformed into **Mel-spectrograms**, followed by feature extraction and model training with **RandomizedSearchCV** for optimal hyperparameter tuning.

---

## 🚀 Project Overview

- **DataSet**: https://www.kaggle.com/competitions/shl-intern-hiring-assessment/data
- **Input**: Audio files (`.wav`) of spoken language samples
- **Output**: Grammar score (range: 0.0 to 5.0, in 0.5 increments)
- **Core Models**: 
  - Random Forest Regressor
  - XGBoost Regressor
  - LightGBM Regressor
- **Cross-validation**: 3-fold CV with extensive hyperparameter tuning
- **Evaluation Metrics**:
  - 📌 **Pearson Correlation**
  - 📉 **Mean Absolute Error (MAE)**
  - 📉 **Root Mean Squared Error (RMSE)**

---

## 🧠 Pipeline

1. **Preprocessing**:
   - Load `.wav` files using `librosa`
   - Convert to **Mel-spectrograms**
   - Normalize and resize to consistent shape
   - Flatten spectrograms to 1D vectors for ML models

2. **Feature Engineering**:
   - Optional dimensionality reduction using PCA (if needed)
   - Rescale features if required for better convergence

3. **Model Training**:
   - Hyperparameter tuning via `RandomizedSearchCV` (`n_iter=50`)
   - Models evaluated on a validation split (80/20)

4. **Evaluation & Selection**:
   - The best model is selected based on **highest Pearson correlation**
   - Performance metrics are plotted for visual validation

---

## 🔍 Hyperparameters Tuned

### XGBoost
- `n_estimators`, `max_depth`, `learning_rate`, `subsample`
- `colsample_bytree`, `gamma`, `reg_alpha`, `reg_lambda`

### Random Forest
- `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
- `max_features`, `bootstrap`

### LightGBM
- `n_estimators`, `learning_rate`, `num_leaves`, `max_depth`
- `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`

---

## 📈 Results

| Model         | Pearson ↑ | MAE ↓ | RMSE ↓ |
|---------------|-----------|-------|--------|
| XGBoost       | 0.70+     | 0.40  | 0.55   |
| LightGBM      | 0.66+     | 0.42  | 0.58   |
| Random Forest | 0.73+     | 0.45  | 0.60   |

*Note: Scores vary depending on tuning and hardware.*

---



---

## 🛠️ Setup & Installation

```bash
# Clone repository
git clone https://github.com/your-repo/grammar-scoring-engine.git
cd grammar-scoring-engine
```

```bash
# Install dependencies
pip install -r requirements.txt
```


## 🧑‍💻 Authors:
Lalith Prabhasith B, CMR College of Engineering & Technology
Specialization: Computer Science & Engineering with AIML

