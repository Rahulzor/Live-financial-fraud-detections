# Real-Time Fraud Detection in Financial Transactions

## 📌 Project Overview
This project implements a **real-time fraud detection system** for financial transactions using **Graph Neural Networks (GNN)** and **LightGBM**.  
It analyzes live blockchain transaction data, extracts graph-based and statistical features, and assigns a **risk score** to detect potentially fraudulent activities.

The system is designed as an **interactive Streamlit dashboard** with advanced visualizations for analysis and monitoring.

---

## 🎯 Objectives
- Detect fraudulent transactions in real time
- Combine graph-based learning with machine learning models
- Visualize transaction risk patterns effectively
- Apply data science and big data concepts to blockchain analytics

---

## 🛠️ Technologies Used
- **Python**
- **Machine Learning (LightGBM)**
- **Graph Neural Networks (GNN)**
- **PyTorch & PyTorch Geometric**
- **NetworkX**
- **Streamlit**
- **Pandas, NumPy, Scikit-learn**
- **Blockchain Data (Ethereum – API based)**

---

## 🧠 Methodology
1. Fetch live transaction data from blockchain
2. Construct transaction graphs from sender–receiver relationships
3. Generate graph embeddings using GNN
4. Perform feature engineering on transaction attributes
5. Train LightGBM classifier for fraud prediction
6. Combine GNN and ML scores to compute final risk score
7. Visualize results using interactive dashboards

---

## ⚙️ Key Features
- Real-time transaction monitoring
- Hybrid fraud detection (GNN + ML)
- Risk score computation
- High-risk transaction identification
- Interactive visualizations:
  - Risk distribution
  - Transaction network graph
  - Timeline analysis
  - Model score comparison

---

## 📊 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score

These metrics are dynamically computed during model evaluation.

---

## 📁 Project Structure
