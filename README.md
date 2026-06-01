
---

```markdown
# 🚖 Uber Fare Prediction (PyTorch)

## 📌 Overview
Deep‑learning regression model predicting Uber ride fares using engineered trip features. Includes data cleaning, scaling, PCA, and a fully connected neural network.

## 🧠 Model
- Input: PCA‑reduced features  
- Architecture: 128 → 64 → 32 → 16 → 1 (ReLU)  
- Loss: MAE  
- Optimizer: Adam (1e‑5)  
- Batch size: 50  
- Device: GPU  

## ⚙️ Pipeline
1. Clean invalid fares  
2. Standard scaling  
3. PCA (95% variance)  
4. Train/Val/Test split  
5. Train PyTorch model  

## ▶️ Usage
```bash
python train.py
python evaluate.py
```

## 📦 Requirements
```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn
```

## 👨‍💻 Author
Tewodros Bewuket
```

---

If you want it **even shorter** or **more aesthetic**, I can compress it further.
