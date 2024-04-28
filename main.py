# Imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS
import numpy as np
# %matplotlib inline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split

# # Mounting Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# Loading Dataset
df = pd.read_csv('./files/dataset.csv')
df.head()

# Checking dataset shape
df.shape

# Checking for missing values
print(df.isnull().sum())

# Counting occurrences of each language
df["language"].value_counts()

# Visualizing distribution of languages before removing duplicates
plt.figure(figsize=(7,5))
sns.countplot(y="language", data=df, hue="language", palette="tab10", legend=False)
plt.show()

# Removing duplicate rows based on 'Text' column
df = df.drop_duplicates(subset='Text')
df = df.reset_index(drop=True)

# Visualizing distribution of languages after removing duplicates
plt.figure(figsize=(7,5))
sns.countplot(y="language", data=df, hue="language", palette="tab10", legend=False)
plt.show()

# Creating a word cloud for English text
english_text_df = df[df['language'] == 'English']
stopwords = set(STOPWORDS)
text2 = "  ".join(review for review in english_text_df['Text'])
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="skyblue", stopwords=stopwords).generate(text2)
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Encoding target variable
le = LabelEncoder()
df["language"]=le.fit_transform(df["language"])

# Displaying unique encoded languages
df["language"].unique()

# Decoding encoded languages
decoded_languages = le.inverse_transform(df["language"])
df["decoded_language"] = decoded_languages
unique_decoded_languages = sorted(df["decoded_language"].unique())
unique_languages = sorted(df["language"].unique())

print("Unique Decoded Languages (Ascending Order):", unique_decoded_languages)
print("Unique Languages (Ascending Order):", unique_languages)

# Displaying DataFrame's first few rows
df.head()

# Calculating average text length
total_length = sum(len(text) for text in df["Text"])
num_texts = len(df["Text"])
average_length = total_length / num_texts
print("Average text length:", average_length)

"""# Data Preprocessing (Cleaning)"""

# Downloading NLTK stopwords
import nltk
nltk.download('stopwords')

# Preprocessing text data
from nltk.corpus import stopwords
import re
import unicodedata
from bs4 import BeautifulSoup

def clean_text(text):
    # Remove HTML tags if present
    if "<" in text:
        text = BeautifulSoup(text, 'html.parser').get_text()

    # Remove URL addresses
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove accented characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    # Remove punctuation
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)

    # Remove irrelevant characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    text = ' '.join(filtered_words)

    return text

# Applying text preprocessing
df["Text"] = df["Text"].apply(clean_text)

# Displaying preprocessed data
df.head()

# Feature Engineering
X = df["Text"]
Y = df["language"]

# Transforming text data into TF-IDF vectors
tf = TfidfVectorizer()
train_data = tf.fit_transform(X)
print(train_data)

# Splitting data into training and testing sets (80% training, 20% testing)
X_train, X_test, Y_train, Y_test = train_test_split(train_data, Y, test_size=0.2, random_state=42)

# Printing total train and test values
print("Total samples in training set (X_train, Y_train):", X_train.shape[0], Y_train.shape[0])
print("Total samples in testing set (X_test, Y_test):", X_test.shape[0], Y_test.shape[0])

# Model Building
svm = SVC()
svm.fit(X_train, Y_train)

# Model Evaluation
y_pred = svm.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

"""# Confusion Matrix"""

# Confusion Matrix
cf = confusion_matrix(Y_test, y_pred)
label_name = unique_decoded_languages

# Visualizing Confusion Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(cf, annot=True, fmt="d", xticklabels=label_name, yticklabels=label_name, cmap="gnuplot", linewidths=3, linecolor='navy')
plt.title("Confusion Matrix", fontsize=20, color="red")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

"""# Classification Report"""

# Classification Report
print(classification_report(Y_test, y_pred, target_names=label_name))
print(f"Overall Accuracy: {accuracy * 100:.2f}%")

# Extracting classification report into a dictionary
classification_report_dict = classification_report(Y_test, y_pred, target_names=label_name, output_dict=True)

# Creating lists to store precision, recall, and F1-score for each class
precision = [classification_report_dict[label]['precision'] for label in label_name]
recall = [classification_report_dict[label]['recall'] for label in label_name]
f1_score = [classification_report_dict[label]['f1-score'] for label in label_name]

# Plotting the scores
plt.figure(figsize=(10, 6))
x = np.arange(len(label_name))
bar_width = 0.2
plt.bar(x, precision, width=bar_width, label='Precision', color='skyblue')
plt.bar(x + bar_width, recall, width=bar_width, label='Recall', color='orange')
plt.bar(x + 2*bar_width, f1_score, width=bar_width, label='F1-Score', color='green')
plt.xlabel('Languages')
plt.ylabel('Scores')
plt.title('Classification Report Metrics')
plt.xticks(x + bar_width, label_name, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

"""# Roc Auc"""

from yellowbrick.classifier import ROCAUC

# Creating SVC model & ROCAUC visualizer.
model = SVC()
visualizer = ROCAUC(model, classes=label_name)

# Visualizer: Fit, score, and display.
visualizer.fit(X_train, Y_train)
visualizer.score(X_test, Y_test)
plt.figure(figsize=(30, 15))
visualizer.show()

"""# Naive Bayes"""

from sklearn.naive_bayes import MultinomialNB

# Fitting Multinomial Naive Bayes (NB) model.
nb = MultinomialNB()
nb.fit(X_train,Y_train)

# Model Evaluation
pred1 = nb.predict(X_test)
accuracy1 = accuracy_score(Y_test, pred1)
print(f"Accuracy: {accuracy1 * 100:.2f}%")

# Confusion Matrix
cf0 = confusion_matrix(Y_test, pred1)
label_name = unique_decoded_languages

# Visualizing Confusion Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(cf0, annot=True, fmt="d", xticklabels=label_name, yticklabels=label_name, cmap="gnuplot", linewidths=3, linecolor='navy')
plt.title("Confusion Matrix", fontsize=20, color="red")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print(classification_report(Y_test, pred1, target_names=label_name))
print(f"Overall Accuracy: {accuracy * 100:.2f}%")

# Extracting classification report into a dictionary
classification_report_dict = classification_report(Y_test, pred1, target_names=label_name, output_dict=True)

# Creating lists to store precision, recall, and F1-score for each class
precision = [classification_report_dict[label]['precision'] for label in label_name]
recall = [classification_report_dict[label]['recall'] for label in label_name]
f1_score = [classification_report_dict[label]['f1-score'] for label in label_name]

# Plotting the scores
plt.figure(figsize=(10, 6))
x = np.arange(len(label_name))
bar_width = 0.2
plt.bar(x, precision, width=bar_width, label='Precision', color='skyblue')
plt.bar(x + bar_width, recall, width=bar_width, label='Recall', color='orange')
plt.bar(x + 2*bar_width, f1_score, width=bar_width, label='F1-Score', color='green')
plt.xlabel('Languages')
plt.ylabel('Scores')
plt.title('Classification Report Metrics')
plt.xticks(x + bar_width, label_name, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Creating MultinomialNB model & ROCAUC visualizer.
model = MultinomialNB()
visualizer = ROCAUC(model, classes=label_name)

# Visualizer: Fit, score, and display.
visualizer.fit(X_train, Y_train)
visualizer.score(X_test, Y_test)
plt.figure(figsize=(30, 15))
visualizer.show()

"""# CountVectorizer"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Feature Engineering
X_data = df["Text"]
Y_data = df["language"]

# Transforming text data Count vectors
vectorizer = CountVectorizer()
new_data = vectorizer.fit_transform(X_data)

# Splitting data into training and testing sets (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(new_data, Y_data, test_size=0.2, random_state=42)

# Printing total train and test values
print("Total samples in training set (x_train, y_train):", x_train.shape[0], y_train.shape[0])
print("Total samples in testing set (x_test, y_test):", x_test.shape[0], y_test.shape[0])

# Model Building
new_svm = SVC()
new_svm.fit(x_train, y_train)

# Model Evaluation
new_pred = new_svm.predict(x_test)
accuracy = accuracy_score(y_test, new_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cf2 = confusion_matrix(y_test, new_pred)
label_name = unique_decoded_languages

# Visualizing Confusion Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(cf2, annot=True, fmt="d", xticklabels=label_name, yticklabels=label_name, cmap="gnuplot", linewidths=3, linecolor='navy')
plt.title("Confusion Matrix", fontsize=20, color="red")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print(classification_report(y_test, new_pred, target_names=label_name))
print(f"Overall Accuracy: {accuracy * 100:.2f}%")

# Creating SVC model & ROCAUC visualizer.
new_model = SVC()
visualizer = ROCAUC(new_model, classes=label_name)

# Visualizer: Fit, score, and display.
visualizer.fit(x_train, y_train)
visualizer.score(x_test, y_test)
plt.figure(figsize=(30, 15))
visualizer.show()

"""# Naive Bayes + Countvectorizer"""

# Fitting Multinomial Naive Bayes (NB) model.
new_nb = MultinomialNB()
new_nb.fit(x_train, y_train)

# Model Evaluation
pred3 = nb.predict(x_test)
accuracy1 = accuracy_score(y_test, pred3)
print(f"Accuracy: {accuracy1 * 100:.2f}%")

# Confusion Matrix
cf4 = confusion_matrix(y_test, pred3)
label_name = unique_decoded_languages

# Visualizing Confusion Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(cf4, annot=True, fmt="d", xticklabels=label_name, yticklabels=label_name, cmap="gnuplot", linewidths=3, linecolor='navy')
plt.title("Confusion Matrix", fontsize=20, color="red")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print(classification_report(y_test, pred3, target_names=label_name))
print(f"Overall Accuracy: {accuracy1 * 100:.2f}%")

# Creating MultinomialNB model & ROCAUC visualizer.
new_mo = MultinomialNB()
visualizer = ROCAUC(new_mo, classes=label_name)

# Visualizer: Fit, score, and display.
visualizer.fit(x_train, y_train)
visualizer.score(x_test, y_test)
plt.figure(figsize=(30, 15))
visualizer.show()