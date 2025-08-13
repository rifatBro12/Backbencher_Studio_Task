# Movie_review_system

# 🎬 Sentiment Analysis on IMDb Reviews

This project performs binary sentiment classification (Positive/Negative) on IMDb movie reviews using both Logistic Regression , Naive Bayes and a Deep Learning model. It includes data preprocessing, TF-IDF vectorization, model training, evaluation, hyperparameter tuning, and deployment-ready prediction functionality.

---

## 📁 Project Structure

- `train_data (1).csv` — Raw dataset containing movie reviews and sentiment labels.
- `logistic_regression_imdb_optimized.pkl` — Saved optimized Logistic Regression model.
- `tfidf_vectorizer.pkl` — Saved TF-IDF vectorizer.
- `predict_sentiment()` — Function to classify new input text.
- `README.md` — Project documentation (this file).

---

## 🧪 Dependencies

Install the required packages using pip:

```bash
pip install pandas scikit-learn nltk beautifulsoup4 seaborn matplotlib tensorflow joblib

import nltk
nltk.download('stopwords')

🧼 Data Preprocessing
Each review is cleaned using the following steps:
- Remove HTML tags
- Convert to lowercase
- Remove numbers and punctuation
- Remove stopwords
- Normalize whitespace

📊 Feature Extraction
TF-IDF vectorization is applied to convert text into numerical features:

🤖 Models
1. Logistic Regression
- Trained with max_iter=200
- Hyperparameter tuning via GridSearchCV
- Saved using joblib

2. Naive Bayes (MultinomialNB)
- Simple and fast probabilistic classifier
- Trained using TF-IDF features
- Evaluation includes accuracy and classification report

3. Support Vector Machine (LinearSVC)
- Effective for high-dimensional spaces
- Trained using TF-IDF features
- Evaluation includes accuracy and classification report

4. Deep Learning (Keras)
- Architecture:
Dense(128) → Dropout → Dense(64) → Dropout → Dense(1)
- Loss: binary_crossentropy
- Optimizer: adam
- Early stopping on validation loss


✅ Model Accuracy Scores:
- Logistic Regression Accuracy: 89.16%
- Precision: 0.89
- Recall: 0.89
- F1-Score: 0.89
- Naive Bayes Accuracy: 85.94%
- Precision: 0.86
- Recall: 0.86
- F1-Score: 0.86
- LinearSVC Accuracy: 87.96%
- Precision: 0.88
- Recall: 0.88
- F1-Score: 0.88
- Deep Learning (Keras) Accuracy: 86.94%
- Precision: 0.87
- Recall: 0.87
- F1-Score: 0.87



📈 Evaluation
 models are evaluated using:
- Accuracy
- Classification Report
- Confusion Matrix (visualized with seaborn)

🔮 Prediction Script
Interactive CLI for real-time sentiment prediction:
while True:
    user_input = input("Enter sentence (or type 'exit' to quit):\n")
    if user_input.lower() == 'exit':
        break
    print("Sentiment:", predict_sentiment(user_input))










