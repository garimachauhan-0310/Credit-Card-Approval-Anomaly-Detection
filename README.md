# 🏦 Hybrid Machine Learning Model for Credit Card Approval with Anomaly Detection

<div align="center">

![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![ML](https://img.shields.io/badge/Machine%20Learning-Hybrid%20Model-purple?style=for-the-badge)

*A smart, explainable, and anomaly-aware credit card approval system powered by a hybrid ML pipeline.*

</div>

---

## 📌 Overview

**FinSight AI** is an end-to-end machine learning system designed to automate and enhance credit card approval decisions. It combines multiple classification models into a **hybrid scoring engine**, integrates **anomaly detection** to flag suspicious applications, and uses **Explainable AI (XAI)** techniques to make every decision transparent and interpretable.

This project goes beyond a simple binary classifier — it introduces a **risk categorization framework**, a **hybrid scoring system**, and powerful **visualizations** to support real-world financial decision-making.

---

## ❗ Problem Statement

Traditional credit card approval systems are:
- ❌ Black-box and non-transparent
- ❌ Unable to detect fraudulent or anomalous applications
- ❌ Rigid — relying on a single model with no ensemble intelligence
- ❌ Lacking explainability for applicants and auditors

**FinSight AI** solves these problems by building a pipeline that is:
- ✅ Hybrid (multi-model ensemble)
- ✅ Anomaly-aware (KMeans + distance-based detection)
- ✅ Risk-stratified via a fusion scoring layer
- ✅ Fully explainable with SHAP

---

## 🛠️ Technologies Used

| Category | Tools & Libraries |
|---|---|
| 🐍 Language | Python 3.9+ |
| 📊 Data Processing | Pandas, NumPy |
| 🤖 Machine Learning | Scikit-learn |
| 📐 Preprocessing | LabelEncoder, StandardScaler, MinMaxScaler |
| 🔍 Anomaly Detection | KMeans (Scikit-learn) + SciPy `cdist` |
| 💡 Explainable AI | SHAP |
| 📈 Visualization | Matplotlib, Seaborn |
| 📓 Environment | Jupyter Notebook |
| 🗂️ Version Control | Git & GitHub |

---

## 🤖 ML Models Used

The following models are trained and combined in the hybrid pipeline:

- 🌲 **Random Forest Classifier** — Ensemble tree-based classifier; also used for feature importance
- 🌿 **Decision Tree Classifier** — Interpretable baseline tree model
- 📐 **Logistic Regression** — Linear probabilistic classifier
- 🔵 **K-Nearest Neighbors (KNN)** — Distance-based classification
- 🔗 **Fusion / Hybrid Layer** — Combines approval probability + anomaly score into a final decision score

---

## 📂 Dataset

### 📊 Source
```text
https://raw.githubusercontent.com/sharmaroshan/Credit-Card-Fraud-Detection/refs/heads/master/Credit_Card_Applications.csv
```
- 🔢 **Features Include:** Credit history, income, employment status, debt, demographics, and more
- 🎯 **Target Variable:** Approved (1) / Rejected (0)
- 🧹 **Preprocessing:**
  - Missing value imputation — mean for numerical, mode for categorical
  - Label Encoding for categorical variables
  - StandardScaler / MinMaxScaler for feature normalization

---

## 🔄 Project Workflow

```
Raw Data
   │
   ▼
Data Preprocessing & Feature Engineering
(Imputation → Label Encoding → Scaling)
   │
   ▼
Model Training (RF, Decision Tree, Logistic Regression, KNN)
   │
   ▼
Anomaly Detection Layer
(KMeans Clustering + Centroid Distance Scoring)
   │
   ▼
Hybrid Fusion Layer
(Approval Probability + Anomaly Score → Final Score)
   │
   ▼
Explainable AI (SHAP Beeswarm + Feature Importance)
   │
   ▼
Final Decision + Visualizations
```

---

## 🏋️ Model Training

- Each classifier is trained independently on the preprocessed dataset
- Models evaluated using accuracy, classification report, confusion matrix, and ROC-AUC
- **Random Forest** is additionally used to extract **feature importance scores**
- The best model's approval probability feeds into the hybrid scoring layer

---

## 📏 Evaluation Metrics

| Metric | Purpose |
|---|---|
| ✅ Accuracy | Overall correctness |
| 🎯 Precision | Avoid false approvals |
| 🔁 Recall | Catch all true approvals |
| ⚖️ F1-Score | Balance of precision & recall |
| 📉 ROC-AUC | Discriminative ability of the model |
| 📊 Confusion Matrix | Breakdown of TP, FP, TN, FN |

---

## 🚨 Anomaly Detection

A **clustering-based unsupervised approach** is used — no labeled anomaly data required:

- 📍 **KMeans Clustering** — Groups applicants into clusters based on feature similarity
- 📏 **Centroid Distance (SciPy `cdist`)** — Computes each applicant's distance from their nearest cluster centroid
- 🚩 Applicants in the **top 5% of distances** are flagged as anomalies
- The anomaly score is **normalized (0–1)** and incorporated into the final hybrid score

---

## 🔀 Hybrid Scoring System

The hybrid scoring system fuses supervised and unsupervised signals:

```
Final Score = 0.7 × Approval Probability
            + 0.3 × (1 − Normalized Anomaly Score)
```

- **Approval Probability** — Output of the trained classifier
- **Anomaly Penalty** — High anomaly score reduces final score
- **Decision Threshold** — Applicants with `final_score > 0.5` → ✅ Approved
- Applicants are also ranked by final score for risk prioritization

---

## 🎨 Risk Categorization

Applicants are ranked and categorized based on their hybrid final score:

| 🏷️ Category | Condition | Decision |
|---|---|---|
| 🟢 Approved | `final_score > 0.5` | ✅ Approved |
| 🔴 Rejected | `final_score ≤ 0.5` | ❌ Rejected |
| 🚩 Anomalous | Top 5% centroid distance | ⚠️ Flagged for Review |

---

## 💡 Explainable AI

- 🔵 **SHAP (SHapley Additive exPlanations)**
  - Applied on the **Random Forest** model
  - **Beeswarm plot** shows global feature impact across all test applicants
  - Reveals which features push approval probability up or down for each individual

- 📊 **Feature Importance Bar Chart**
  - Extracted from Random Forest's `feature_importances_`
  - Visualized as a horizontal bar chart for easy interpretation

> *Explainability ensures fairness, regulatory compliance, and trust in the system.*

---

## 📊 Visualizations Included

- 📊 **Confusion Matrices** — Per model, using `ConfusionMatrixDisplay`
  <img width="498" height="455" alt="image" src="https://github.com/user-attachments/assets/33225ea7-4828-4718-8ab3-1d3efcc79953" />
  <img width="498" height="455" alt="image" src="https://github.com/user-attachments/assets/1d417173-2758-4345-90e1-e5010b262378" />
  <img width="498" height="455" alt="image" src="https://github.com/user-attachments/assets/2a70b49e-41a8-4aaa-b3c2-04009b364eb6" />

- 📉 **ROC Curve** — With AUC score plotted per classifier
  <img width="547" height="435" alt="image" src="https://github.com/user-attachments/assets/9d29322a-18c4-4af4-8b8a-7a33d7c39a66" />
  
- 📈 **Feature Importance** — Horizontal bar chart from Random Forest
  <img width="607" height="435" alt="image" src="https://github.com/user-attachments/assets/3ec69377-f3c0-43fb-89c4-5c8bd29bdc0c" />
  
- 🐝 **SHAP Beeswarm Plot** — Global feature explanation across test samples
  <img width="857" height="497" alt="image" src="https://github.com/user-attachments/assets/9da6c82d-d2c8-4766-a0db-e6d6f5dc8ccd" />

---

## ⭐ Key Features of This Project

- 🔀 **Hybrid Ensemble** — Multiple ML models combined for robust decisions
- 🚨 **Clustering-Based Anomaly Detection** — KMeans + distance scoring, no labels needed
- 🧮 **Fusion Scoring Layer** — Mathematically combines approval probability and anomaly penalty
- 💡 **Explainable AI** — SHAP beeswarm + feature importance for full interpretability
- 📊 **Rich Visualizations** — ROC curves, confusion matrices, SHAP plots, and more
- 🧹 **Clean Preprocessing** — Handles missing values, encoding, and scaling end-to-end
- 📦 **Single Notebook** — Entire pipeline from raw data to final decision in one place

---

## 🗂️ Repository Structure

```
📦 credit-card-approval-hybrid-ml/
│
├── 📁 data/
│   └── credit_card_data.csv       # Dataset used for training
│
├── 📓 Credit_card_approval_&_anomaly_detection.ipynb    # Main notebook — full pipeline
│
├── 📄 requirements.txt
├── 📄 LICENSE
└── 📄 README.md
```

---

## 🤝 Contributions & Suggestions

Contributions, feature suggestions, and improvements are highly welcome!

If you have ideas to enhance **FinSight AI** — whether related to:

- 📐 Investment logic and risk modeling
- 💡 Explainability and transparency
- 🎨 UI/UX improvements
- ⚡ Performance or scalability
- 🔌 New features or integrations

...please feel free to contribute!

### How to Contribute

1. 🍴 **Fork** the repository
2. 🌿 **Create a new branch** (`feature/your-feature-name`)
3. 💬 **Commit your changes** with clear messages
4. 📬 **Open a Pull Request** describing your changes

You may also:
- 🐛 Open an **Issue** for bug reports or feature requests
- 💭 Share **feedback** or architectural suggestions

> All contributions should aim to keep the project **ethical**, **explainable**, and **user-focused**.

---

⭐ **If you find this project useful, consider giving it a star!** It helps others discover the project and motivates further development.

---

## 🧾 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🧑‍💻 Author

**Garima Chauhan**

- 🎓 B.Tech CSE (AIML) @ Jaypee University of Information Technology, Solan
- 📍 From Noida, India
- 📫 [garimachauhan03102006@gmail.com](mailto:garimachauhan03102006@gmail.com)
- 🔗 [LinkedIn](#) | [GitHub](#)

---

<div align="center">

*Built with ❤️ for fair, transparent, and intelligent credit decisions.*

</div>
