# Sentiment Analysis on IMDB Reviews

**Author**: [Md. Mahmudul Hasan]  

This project performs sentiment analysis on IMDB movie reviews using machine learning and deep learning techniques. The dataset contains movie reviews labeled as positive (1) or negative (0). The goal is to preprocess the text data, train classification models, and predict the sentiment of new reviews.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Preprocessing](#preprocessing)
- [Models](#models)
- [Evaluation](#evaluation)
- [Saving and Loading Models](#saving-and-loading-models)
- [Predicting Sentiment](#predicting-sentiment)
- [Results](#results)
- [Visualization](#visualization)
- [License](#license)

## Project Overview
This project processes IMDB movie reviews to classify them as positive or negative. It includes text preprocessing, feature extraction using TF-IDF, and training multiple models, including Logistic Regression, Naive Bayes, Linear SVM, and a Deep Learning model. Hyperparameter tuning is performed using GridSearchCV, and the models are evaluated based on accuracy and classification metrics.

## Dataset
The dataset (`train_data.csv`) contains 25,000 movie reviews with two columns:
- `review`: The text of the movie review.
- `label`: The sentiment label (0 for negative, 1 for positive).

The dataset is split into 80% training (20,000 samples) and 20% testing (5,000 samples).

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies
The following Python libraries are required:
- `pandas`
- `scikit-learn`
- `beautifulsoup4`
- `nltk`
- `matplotlib`
- `seaborn`
- `tensorflow`
- `joblib`
- `streamlit`

Install them using:
```bash
pip install pandas scikit-learn beautifulsoup4 nltk matplotlib seaborn tensorflow joblib
```

Additionally, download NLTK stopwords:
```python
import nltk
nltk.download('stopwords')
```

## Usage
1. Place the `train_data.csv` file in the project directory.
2. Run the script to preprocess data, train models, and evaluate performance:
   ```bash
   python sentiment_analysis.py
   ```
3. Use the `predict_sentiment` function to classify new reviews (see [Predicting Sentiment](#predicting-sentiment)).

## Preprocessing
Text preprocessing steps include:
- Removing HTML tags using `BeautifulSoup`.
- Converting text to lowercase.
- Removing numbers and punctuation.
- Removing extra spaces.
- Removing stopwords using NLTK's English stopwords.
- Saving the cleaned dataset to `df_clean.csv`.

The cleaned text is transformed into numerical features using `TfidfVectorizer` with a maximum of 5,000 features.

## Models
The project trains and evaluates the following models:
1. **Logistic Regression**:
   - Initial model: Accuracy = 89.16%.
   - Hyperparameter-tuned model (GridSearchCV with `C`, `solver`, `max_iter`): Best parameters `{'C': 1, 'max_iter': 200, 'solver': 'liblinear'}` with accuracy = 89.10%.
2. **Naive Bayes (MultinomialNB)**:
   - Hyperparameter-tuned model (GridSearchCV with `alpha`, `fit_prior`): Best parameters `{'alpha': 1.5, 'fit_prior': True}` with accuracy = 86.04%.
3. **Linear SVM (LinearSVC)**:
   - Accuracy = 87.96%.
4. **Deep Learning (TensorFlow/Keras)**:
   - Neural network with two hidden layers (128 and 64 units, ReLU activation), dropout (0.3), and sigmoid output.
   - Trained with early stopping to prevent overfitting.
   - Accuracy = 88.30%.

## Evaluation
Models are evaluated using:
- **Accuracy**: Percentage of correct predictions.
- **Classification Report**: Precision, recall, and F1-score for both classes (Negative and Positive).
- **Confusion Matrix**: Visualized using a heatmap for the Logistic Regression model.

### Results Summary
| Model                | Accuracy (%) | Precision (0/1) | Recall (0/1) | F1-Score (0/1) |
|----------------------|--------------|-----------------|--------------|----------------|
| Logistic Regression  | 89.16        | 0.91 / 0.87     | 0.87 / 0.92  | 0.89 / 0.89    |
| Naive Bayes          | 86.04        | 0.87 / 0.85     | 0.85 / 0.87  | 0.86 / 0.86    |
| Linear SVM           | 87.96        | 0.89 / 0.87     | 0.86 / 0.90  | 0.88 / 0.88    |
| Deep Learning        | 88.30        | 0.88 / 0.89     | 0.89 / 0.87  | 0.89 / 0.88    |
| Optimized Logistic   | 89.10        | 0.91 / 0.87     | 0.87 / 0.92  | 0.89 / 0.89    |

## Visualization
A confusion matrix for the Logistic Regression model is visualized using `seaborn` and `matplotlib`. To generate the plot:
```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

## Saving and Loading Models
- The optimized Logistic Regression model and TF-IDF vectorizer are saved as:
  - `logistic_regression_imdb_optimized.pkl`
  - `tfidf_vectorizer.pkl`
- Load them using:
  ```python
  import joblib
  model = joblib.load('logistic_regression_imdb_optimized.pkl')
  vectorizer = joblib.load('tfidf_vectorizer.pkl')
  ```

## Predicting Sentiment
A function `predict_sentiment` is provided to classify new reviews:
```python
def predict_sentiment(text):
    cleaned = clean_text(text)
    vect_text = vectorizer.transform([cleaned])
    pred = model.predict(vect_text)[0]
    return "Positive" if pred == 1 else "Negative"
```

Example usage:
```python
sentence = "I really enjoyed this movie! The plot was exciting and the acting was superb."
print("Predicted Sentiment:", predict_sentiment(sentence))  # Output: Positive

sentence = "The movie was bad."
print("Predicted Sentiment:", predict_sentiment(sentence))  # Output: Negative
```

## License
This project is licensed under the MIT License.






