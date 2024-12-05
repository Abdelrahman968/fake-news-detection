
# Fake News Detection - NLP using NLTK

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction
Fake News Detection is a Natural Language Processing (NLP) project aimed at identifying whether a given news article is real or fake. This project utilizes the NLTK (Natural Language Toolkit) library to preprocess text data and build a machine learning model to classify news articles. The system can be used as a preliminary tool to combat misinformation.

---

## Features
- Preprocessing of text data (tokenization, stemming, and removing stopwords).
- TF-IDF vectorization for feature extraction.
- Training and testing a machine learning model (e.g., Logistic Regression, Naive Bayes).
- User-friendly interface to input and classify news articles.

---

## Technologies Used
- **Python 3.x**
- **NLTK** for text preprocessing
- **Scikit-learn** for machine learning model development
- **Pandas** and **NumPy** for data manipulation
- **Flask** (optional) for building a web interface

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Abdelrahman968/fake-news-detection.git
   cd fake-news-detection
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the NLTK datasets:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

---

## Usage
1. Preprocess the dataset:
   Run the script to preprocess the data and split it into training and testing sets.

2. Train the model:
   ```bash
   run the Project.ipynb
   ```

3. Classify new articles:
   Input the text of the news article to get the classification.

---

## Dataset
The dataset used in this project consists of labeled news articles, where each article is categorized as `Real` or `Fake`. Example datasets:
- [Fake News Dataset from Kaggle](https://www.kaggle.com/c/fake-news/data)

Place the dataset in the `data/` directory.

---

## Project Workflow
1. **Data Preprocessing:**
   - Tokenization
   - Removing stopwords
   - Stemming or lemmatization

2. **Feature Extraction:**
   - TF-IDF Vectorization

3. **Model Development:**
   - Training using algorithms like Logistic Regression or Naive Bayes
   - Hyperparameter tuning

4. **Evaluation:**
   - Confusion matrix
   - Accuracy, Precision, Recall, F1-score

---

## Results
The trained model achieved the following Accuracy results:
- Naive Bayes: 0.9261
- Logistic Regression: 0.9866
- Random Forest: 0.9981
- SVM: 0.9927
- Voting Classifier: 0.9909

The trained model achieved the following results:
- Accuracy: 99.09%
- Precision: 99.00%
- Recall: 99.00%
- F1-score: 99.00%
---

## Future Enhancements
- Expand the dataset to improve model performance.
- Add more sophisticated NLP techniques, such as BERT or other transformers.
- Deploy the model using Flask or FastAPI for a user-friendly web application.
- Implement real-time news scraping for classification.

---

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure your code adheres to the project's style guidelines.

---

## License
This project is licensed under the [MIT License](LICENSE).
