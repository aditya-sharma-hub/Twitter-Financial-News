# 📰 Twitter Financial News Classification  

> 🚀 A Deep Learning–powered NLP project that classifies financial tweets into 20 unique market-related categories using CNN, ANN & RNN architectures.  
> It combines data science, machine learning, and finance to uncover how the world talks about markets on Twitter.

<p align="left"> 
<img src="https://img.shields.io/badge/Domain-Finance%20Analytics-blue?style=for-the-badge"/> 
<img src="https://img.shields.io/badge/Machine%20Learning-Text%20Classification-green?style=for-the-badge"/> 
<img src="https://img.shields.io/badge/Deployment-Streamlit-orange?style=for-the-badge"/> 
</p>

## 📘 Overview  
The **Twitter Financial News Classification** project focuses on building an intelligent text classification pipeline capable of identifying financial tweet topics such as *Earnings, Analyst Updates, Stock Movement,* and *M&A News.*  

This project leverages **Natural Language Processing (NLP)** and **Deep Learning** to interpret financial content and provide insights that can support traders, analysts, and AI-driven finance applications.  

The model was trained and validated on real-world financial tweets to achieve high accuracy and generalization performance.

---

## ❓ Problem Statement  
With millions of finance-related tweets published daily, manual classification is impractical.  
This project aims to automate the categorization of financial tweets — enabling analysts and systems to:  
- Monitor breaking financial news efficiently  
- Detect trends across financial topics  
- Facilitate automated sentiment and topic analysis in real time  

---

## 🎯 Key Objectives  
✅ Preprocess and clean raw Twitter data (remove links, hashtags, punctuation)  
✅ Perform detailed Exploratory Data Analysis (EDA) to identify distribution and imbalance  
✅ Extract numerical features using **CountVectorizer** and **Tokenizer**  
✅ Build and compare **CNN**, **RNN**, and **ANN** architectures  
✅ Achieve high validation accuracy with deep learning techniques  
✅ Provide an explainable and scalable text classification pipeline  

---

## 🧾 Dataset Features  

**Dataset:** Twitter Financial News (21,107 English-language finance tweets)  
**Split:**  
- `train_data.csv` → 16,990 samples  
- `valid_data.csv` → 4,117 samples  

Each tweet is annotated into one of **20 financial categories**:  

| ID | Label | Category |
|----|--------|-----------|
| 0 | LABEL_0 | Analyst Update |
| 1 | LABEL_1 | Fed / Central Banks |
| 2 | LABEL_2 | Company / Product News |
| 3 | LABEL_3 | Treasuries / Corporate Debt |
| 4 | LABEL_4 | Dividend |
| 5 | LABEL_5 | Earnings |
| 6 | LABEL_6 | Energy / Oil |
| 7 | LABEL_7 | Financials |
| 8 | LABEL_8 | Currencies |
| 9 | LABEL_9 | General News / Opinion |
| 10 | LABEL_10 | Gold / Metals / Materials |
| 11 | LABEL_11 | IPO |
| 12 | LABEL_12 | Legal / Regulation |
| 13 | LABEL_13 | M&A / Investments |
| 14 | LABEL_14 | Macro |
| 15 | LABEL_15 | Markets |
| 16 | LABEL_16 | Politics |
| 17 | LABEL_17 | Personnel Change |
| 18 | LABEL_18 | Stock Commentary |
| 19 | LABEL_19 | Stock Movement |

---

## 📊 Analysis Summary  

🧹 **Data Cleaning:**  
Removed URLs, hashtags, special characters, and numbers → normalized text using lemmatization & stopword removal.  

📈 **Exploratory Data Analysis (EDA):**  
- Visualized label imbalance using Seaborn countplots  
- Generated **WordClouds** for most frequent terms  
- Mapped 20 financial topics for distribution overview  

🧩 **Feature Engineering:**  
- Used **CountVectorizer** and **Tokenizer** for numerical representation  
- Managed class imbalance using inverse class weights  

🧠 **Modeling:**  
Implemented and compared:
- **ANN (Feedforward Network)**
- **CNN (Convolutional Neural Network)**
- **RNN (SimpleRNN for sequential text data)**  

🏁 **Best Model:**  
**CNN Model** achieved ~**79% validation accuracy**, outperforming traditional models and demonstrating strong generalization.  

---

## 🧠 Tech Stack  

**Languages & Tools:**  
> 🐍 Python • 📓 Jupyter Notebook • 💻 VS Code  

**Libraries Used:**  
- Pandas | NumPy | Matplotlib | Seaborn  
- NLTK | WordCloud  
- Scikit-learn | TensorFlow | Keras  

**Core Concepts:**  
- Natural Language Processing (NLP)  
- Text Vectorization (CountVectorizer, Tokenizer)  
- Deep Learning (ANN, CNN, RNN)  
- Model Training, Validation & Visualization  

---

**Concepts Used:**  
- NLP preprocessing  
- Tokenization and Embedding  
- Deep Learning Architectures (CNN, ANN, RNN)  
- Multi-class text classification  
- Model Evaluation & Visualization  

---

## ⚙️ How to Run  
1. **Clone the repository:**  
   ```bash
   git clone https://github.com/your-username/Twitter-Financial-News.git
   cd Twitter-Financial-News

---

## 🧠 Author

**Aditya Sharma**  
_Machine Learning Enthusiast | Data Scientist_


