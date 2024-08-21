import os
import zipfile
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Extract the dataset (the zip file to a specific directory)
with zipfile.ZipFile('sentences.zip', 'r') as zip_ref:
    zip_ref.extractall('sentences')

# Read the dataset
df = pd.read_csv('sentences/sentiment labelled sentences/yelp_labelled.txt', delimiter='\t', engine='python', quoting=3, names=['review', 'status'])

print(df.head) 

# Clean the dataset
def clean_data(review):
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    return review

df['review'] = df['review'].apply(clean_data)

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Remove stop words
def remove_stop_words(review):
    stop_words = set(stopwords.words('english'))
    review_words = review.split()
    filtered_words = [word for word in review_words if word not in stop_words]
    return ' '.join(filtered_words)

df['review'] = df['review'].apply(remove_stop_words)

# Initialize the WordNetLemmatizer correctly
lemmatizer = WordNetLemmatizer()

# Lemmatize the text
def lematize(review):
    review_words = review.split()
    review = [lemmatizer.lemmatize(word) for word in review_words]
    return ' '.join(review)

df['review'] = df['review'].apply(lematize)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['review']).toarray()
y = df['status']

# Split the data into training and testing sets
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y, test_size=0.20, random_state=101)

#this was te methos that was used to test the classifiers
    # Initialize classifiers
    #classifiers = [
        #GradientBoostingClassifier(), GaussianNB(), HistGradientBoostingClassifier(),
        #RandomForestClassifier(), LogisticRegression()
    #]

    # Train and evaluate classifiers
    #for classifier in classifiers:
        #classifier.fit(X_train_s, y_train_s)
        #print(f'The {classifier.__class__.__name__} accuracy is {accuracy_score(y_test_s, classifier.predict(X_test_s))}')

# Train and evaluate Logistic Regression separately
classifier = LogisticRegression()
classifier.fit(X_train_s, y_train_s)
predictions = classifier.predict(X_test_s)
conf_matrix = confusion_matrix(y_test_s, predictions)
print(conf_matrix)
print(classification_report(y_test_s, predictions))

#for the text
def preprocess_and_predict(text, model, vectorizer):
    text = clean_data(text)
    text = remove_stop_words(text)
    text = lematize(text)
    text_vectorized = vectorizer.transform([text]).toarray()
    prediction = model.predict(text_vectorized)
    return prediction

new_text = "i love this resturant."
predicted_sentiment = preprocess_and_predict(new_text, classifier, vectorizer)
print(f"The predicted sentiment for the new text is: {predicted_sentiment[0]}")

