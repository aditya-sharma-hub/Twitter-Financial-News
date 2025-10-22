# 🐦 Twitter Financial News — Sentiment Classification using Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/Domain-Finance%20Analytics-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-Text%20Classification-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Deployment-Streamlit-orange?style=for-the-badge"/>
</p>

---

## 🧠 Overview
This project focuses on **analyzing and classifying financial tweets** from Twitter to uncover market sentiment and key financial topics.  
By leveraging **Machine Learning (ML)** and **Deep Learning (DL)** models such as **ANN, CNN, and RNN**, the project aims to categorize financial news into various domains like:

- 🏦 *Markets, IPOs, Central Banks, Legal Updates, Stock Movement, M&A, and more.*

This project helps **financial analysts, investors, and data enthusiasts** to gain **real-time insights** from financial discussions on social media.

---

## 🎯 Objective
- Classify finance-related tweets into **20 financial categories** (like Earnings, Dividends, Stock Movement, etc.).
- Understand how public sentiment on Twitter reflects **market behavior**.
- Develop and deploy a **deep learning-based classifier** that predicts the topic of any financial tweet.

---

## 📊 Dataset Information
**Dataset:** [Twitter Financial News Dataset](https://drive.google.com/drive/folders/1CVYg2mkQ4MYv7ZGbR6V6UmPZDfsyOats?usp=sharing)  
- 📈 Total Records: 21,107 annotated tweets  
- 🏷️ 20 Financial Categories:
  - Analyst Update, Fed | Central Banks, Dividend, Earnings, Legal | Regulation, IPO, Markets, M&A, etc.  
- 💬 Each tweet is labeled based on its financial domain.

---

## 🧩 Techniques & Methodology

### 🧹 **1. Data Preprocessing**
- Removed URLs, hashtags, mentions, numbers, and special characters.
- Converted text to lowercase and applied **lemmatization**.
- Tokenized text using Keras’ Tokenizer.
- Handled **class imbalance** using **inverse class weighting**.

### 📊 **2. Exploratory Data Analysis (EDA)**
- WordCloud visualizations to identify frequent financial terms.
- Count plots showing label distribution.
- Observed heavy imbalance — requiring class-weight adjustments.

### ⚙️ **3. Feature Engineering**
- Text vectorization using **TF-IDF** and **CountVectorizer**.
- Sequence padding for Deep Learning input consistency.

### 🧠 **4. Model Building**
| Model | Type | Accuracy | Remarks |
|--------|-------|----------|----------|
| Logistic Regression | ML | ~53% | Baseline Model |
| CNN (1D) | Deep Learning | ~78.9% | Extracted strong contextual features |
| ANN | Deep Learning | **~83.3%** | Best overall model |
| RNN | Deep Learning | ~77% | Captured sequential context |

### 🚀 **5. Deployment**
- Deployed via **Streamlit** for real-time classification.
- Users can input financial tweets and instantly get the predicted category.

---

## 🧰 Tools & Technologies
| Category | Tools |
|-----------|--------|
| **Language** | Python |
| **Libraries** | pandas, numpy, sklearn, tensorflow, keras, nltk, seaborn, matplotlib, wordcloud, tweepy, streamlit |
| **Environment** | Jupyter Notebook, VS Code |
| **Deployment** | Streamlit |
| **Visualization** | Matplotlib, Seaborn, WordCloud |

---

## 🏗️ Project Structure
