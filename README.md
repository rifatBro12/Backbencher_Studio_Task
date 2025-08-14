#  Sentiment Analysis on IMDb Reviews

This project performs binary sentiment classification (Positive/Negative) on IMDb movie reviews using both Logistic Regression , Naive Bayes,
SVM and a Deep Learning model. It includes data preprocessing, TF-IDF vectorization, model training, evaluation, hyperparameter tuning, and deployment-ready prediction functionality.

---

##  Project Structure

- `train_data.csv` — kaggle dataset containing movie reviews and sentiment labels.
- `logistic_regression_imdb_optimized.pkl` — Saved optimized Logistic Regression model.
- `tfidf_vectorizer.pkl` — Saved TF-IDF vectorizer.
- `predict_sentiment()` — Function to classify new input text.
- `README.md` — Project documentation (this file).

---

##  Dependencies

Install the required packages using pip

```bash
pip install pandas scikit-learn nltk beautifulsoup4 seaborn matplotlib tensorflow joblib

import nltk
nltk.download('stopwords')

  Data Preprocessing

Each review is cleaned using the following steps:
- Remove HTML tags
- Convert to lowercase
- Remove numbers and punctuation
- Remove stopwords
- Normalize whitespace

 Feature Extraction

TF-IDF vectorization is applied to convert text into numerical features

 ## Models

1. Logistic Regression
- Trained with max_iter=200
- Hyperparameter tuning via GridSearchCV
- Saved using joblib

2. Naive Bayes (MultinomialNB)
- Probabilistic classifier optimized via GridSearchCV
- Best parameters: alpha=1.5, fit_prior=True
- Trained using TF-IDF features

3. Support Vector Machine (LinearSVC)
- Effective for high-dimensional spaces
- Trained using TF-IDF features
- Evaluation includes accuracy and classification report

4. Deep Learning (Keras)
- Architecture:
Dense(128, relu) → Dropout(0.3) → Dense(64, relu) → Dropout(0.3) → Dense(1, sigmoid)
- Input: TF-IDF vectors converted to dense arrays
- Loss: binary_crossentropy
- Optimizer: adam
- Early stopping on validation loss (patience=3)
- Trained for up to 20 epochs with batch_size=64
- Evaluation includes accuracy and classification report
- Test Accuracy: Reported via model.evaluate()
- Predictions: Thresholded at 0.5 to convert probabilities to binary labels


 ## Model Accuracy Scores:

- Logistic Regression Accuracy: 89.16%

- Naive Bayes Accuracy: 86.04%

- LinearSVC Accuracy: 87.96%

- Deep Learning (Keras) Accuracy: 88.30%





## Evaluation
- Accuracy Score
Measures overall correctness of predictions.
- Classification Report
Includes precision, recall, and F1-score for each class, giving insight into model balance and bias.
- Confusion Matrix (Visualized with Seaborn)
A visualization that shows true vs. predicted labels, helping identify misclassifications and class-specific performance.

## Prediction Script
A standalone script enables instant sentiment prediction from any given text:
   Loads the saved TF-IDF vectorizer and trained classification model
   Accepts a predefined example sentence (or can be easily adapted for user input)
   Transforms the sentence into TF-IDF features for model compatibility
   Uses the trained model to predict sentiment (Positive / Negative)
   Outputs the prediction result instantly for quick demonstration and validation














