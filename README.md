# 📊 Customer 360 Intelligence — Telecom Machine Learning Workshop

A comprehensive end-to-end Machine Learning and Data Science workflow built for **NileConnect Telecom** to analyze customer behavior, predict churn, forecast revenue, perform advanced segmentation, and detect anomalies.

---

## 🛠️ Project Structure & Architecture
The project is modularized into distinct Jupyter Notebooks covering the complete Data Science lifecycle:
1. **Data Understanding & EDA (`DataUnderstanding.ipynb`)**: Initial data exploration, missing value analysis, and distribution checks.
2. **Churn Classification (`ChurnClassification.ipynb`)**: Building and evaluating supervised machine learning classifiers to predict customer churn risk.
3. **Revenue Regression (`RevenueRegression.ipynb`)**: Estimating 12-month customer revenue using regression models.
4. **Customer Segmentation (`CustomerSegmentation.ipynb`)**: Applying unsupervised learning (**K-Means**, **Agglomerative Clustering**, and **DBSCAN**) using behavioral metrics to identify distinct customer tiers.
5. **Dimensionality Reduction & Anomaly Detection (`PCA_AnomalyDetection.ipynb`)**: Using **PCA** for 2D visualization of clusters and **Isolation Forest** to identify unusual customer patterns/outliers.
6. **Customer 360 Integration (`Customer360Integration.ipynb`)**: Merging all insights, defining retention-priority rules, and exporting the final master dataset.

---

## 🚀 Key Features & Models
* **Supervised Learning**: Classification & Regression models for churn probability and revenue prediction.
* **Unsupervised Learning**: K-Means clustering ($k=4$) backed by Elbow Method and Silhouette Score analysis.
* **Dimensionality Reduction**: Principal Component Analysis (PCA) capturing the core variance of behavioral features.
* **Anomaly Detection**: Isolation Forest to spot atypical user behaviors and edge cases.
* **Business Intelligence**: Custom retention-priority flagging and segment profiling.

---

## ⚙️ Tech Stack
* **Language**: Python
* **Libraries**: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Environment**: Conda (`ml_master`), Jupyter Notebook

---

## 📂 Final Output
The final unified master table (`customer_360_final_output.csv`) combines customer IDs, churn probabilities, predicted revenues, assigned segments, anomaly flags, and retention priority levels to drive data-backed marketing and retention strategies.
