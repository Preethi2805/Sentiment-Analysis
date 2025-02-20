# Sentiment Analysis of Financial News

The goal is to classify the sentiment of financial news into three categories: **Positive**, **Negative**, or **Neutral**. The dataset used is the **Financial PhraseBank** from Kaggle, which contains financial news headlines labeled with their corresponding sentiment.

---

## Project Overview
The project involves:
- **Data Collection**: Downloading and loading the Financial PhraseBank dataset.
- **Data Preprocessing**: Cleaning, tokenizing, and encoding the text data for model training.
- **Model Training**: Building and training machine learning models, including **Logistic Regression** and **LSTM (Long Short-Term Memory)**.
- **Model Evaluation**: Evaluating the performance of the models using accuracy and classification reports.
- **Prediction**: Building a predictive system to classify the sentiment of new financial news headlines.

---

## Dataset
The dataset used in this project is the **Financial PhraseBank** dataset from Kaggle, which contains financial news headlines labeled with one of three sentiments:
- **Positive**: Indicates positive sentiment.
- **Negative**: Indicates negative sentiment.
- **Neutral**: Indicates neutral sentiment.

The dataset contains **4,845 samples** with two columns:
- **News**: The financial news headline.
- **Sentiment**: The sentiment label (Positive, Negative, or Neutral).

---

## Data Preprocessing
- **Handling Missing Values**: The dataset was checked for missing values, and no missing values were found.
- **Encoding Sentiment Labels**: The sentiment labels were encoded as follows:
  - Positive: 1
  - Negative: 0
  - Neutral: 2
- **Tokenization**: The text data was tokenized using the **Tokenizer** class from TensorFlow, which converts words into integers based on their frequency.
- **Padding**: The sequences were padded to ensure uniform length for input into the LSTM model.

---

## Model Training
Two models were trained and evaluated:
1. **Logistic Regression**:
   - Achieved an accuracy of **73.71%**.
   - The classification report showed precision, recall, and F1-score for each sentiment class.

2. **LSTM (Long Short-Term Memory)**:
   - A deep learning model designed for sequential data.
   - Achieved a test accuracy of **72.89%**.
   - The model was trained for 5 epochs with a batch size of 32.

---

## Results
- **Logistic Regression**:
  - **Accuracy**: 73.71%
  - **Classification Report**:
    - Positive: Precision = 0.72, Recall = 0.48, F1-Score = 0.57
    - Negative: Precision = 0.76, Recall = 0.28, F1-Score = 0.41
    - Neutral: Precision = 0.74, Recall = 0.95, F1-Score = 0.83

- **LSTM**:
  - **Test Accuracy**: 72.89%
  - **Test Loss**: 0.946

---

## Technologies Used
- **Python**: Primary programming language.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computations.
- **Scikit-learn**: For Logistic Regression and evaluation metrics.
- **TensorFlow/Keras**: For building and training the LSTM model.
- **Matplotlib & Seaborn**: Data visualization.

---

---

## Future Work
- Experiment with more advanced models like **BERT** or **Transformer-based models** for better accuracy.
- Perform hyperparameter tuning to improve model performance.
