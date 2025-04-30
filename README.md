# email-spam-classifier-new
This project is a machine learning-based web application that classifies SMS text messages as either Spam or Not Spam. It uses Natural Language Processing (NLP) techniques with TF-IDF vectorization and a Naive Bayes classifier to make predictions.

It’s built with Python and deployed using Streamlit for a smooth and interactive user interface. The goal is to demonstrate how ML can be applied to real-world problems like filtering spam in communication systems.

🔍 Detailed Explanation of Project Workflow
📥 1. Data Collection
Dataset Used: SMS Spam Collection Dataset

Structure:

Two columns: label (spam or ham), and message (the actual text).

5,572 labeled SMS messages.

🧹 2. Data Preprocessing (in train.py)
Preprocessing is critical to clean and convert raw text into a format suitable for training.

Steps performed:

Convert to lowercase

Remove punctuation and special characters

Tokenize and remove stopwords

Lemmatize words

📄 This is implemented using nltk, re, and sklearn.

🔠 3. Text Vectorization using TF-IDF
Before feeding the data into a machine learning model, we convert text to numbers using TF-IDF (Term Frequency–Inverse Document Frequency).

Why TF-IDF?
It gives importance to rare but significant words rather than frequent but less meaningful ones like "the", "is", etc.

🤖 4. Model Training
Algorithm Used: Multinomial Naive Bayes

Why? It's fast and performs well for text classification.

Training Accuracy: Typically ~96–98% (depending on data split and cleaning)

🔍 You save the trained model as model.pkl and the vectorizer as vectorizer.pkl using joblib.

💻 5. Streamlit Frontend (app.py)
A simple, interactive UI using Streamlit allows users to:

Input an SMS message.

Press a button to classify it.

Instantly get output as “Spam” or “Not Spam”.

⚙️ Technologies Used
 • Python

 • Streamlit

 • Scikit-learn

 • Pandas

 • Numpy

 • Joblib

 • TF-IDF (Text vectorization)

 • Naive Bayes (Classification model)
