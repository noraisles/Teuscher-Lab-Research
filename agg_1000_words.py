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

# Defines paths for extracting dataset
zip_file_path = 'A Comprehensive Dataset for Automated Cyberbullying Detection (2).zip'
extracted_dir = 'extracted_data'

# Extract the ZIP file to the specified directory
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_dir)

# Verify the contents of the extracted directory
extracted_contents = os.listdir(extracted_dir)
print("Extracted contents:", extracted_contents)

# Assuming the extracted directory contains a subdirectory
subdir = extracted_contents[0]  # This would be 'A Comprehensive Dataset for Automated Cyberbullying Detection (2)'
csv_file_name = '5. Communication_Data_Among_Users.csv'
csv_file_path = os.path.join(extracted_dir, subdir, csv_file_name)

# Read the CSV file
df = pd.read_csv(csv_file_path, nrows=1000, delimiter=',', engine='python', quoting=3, skiprows=1, usecols=[4, 5], names=['review', 'status'])

# Print the first few rows of the DataFrame to verify
print(df.head())

# Drop rows with missing values in the 'review' or 'status' column
df.dropna(subset=['review', 'status'], inplace=True)

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
def lemmatize(review):
    review_words = review.split()
    review = [lemmatizer.lemmatize(word) for word in review_words]
    return ' '.join(review)

df['review'] = df['review'].apply(lemmatize)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['review']).toarray()
y = df['status']

# Verify if there are still any NaN values in the target variable 'status'
if y.isnull().sum() > 0:
    print("Warning: Missing values found in 'status' column.")
else:
    print("No missing values found in 'status' column.")

# Split the data into training and testing sets
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y, test_size=0.20, random_state=101)

# Train and evaluate Logistic Regression separately
classifier = LogisticRegression()
classifier.fit(X_train_s, y_train_s)
predictions = classifier.predict(X_test_s)
conf_matrix = confusion_matrix(y_test_s, predictions)
print(conf_matrix)
print(classification_report(y_test_s, predictions))

# For the text preprocessing and prediction
def preprocess_and_predict(text, model, vectorizer):
    text = clean_data(text)
    text = remove_stop_words(text)
    text = lemmatize(text)
    text_vectorized = vectorizer.transform([text]).toarray()
    prediction = model.predict(text_vectorized)
    return prediction

new_text = input("What is the text you would like to analize? ")
predicted_sentiment = preprocess_and_predict(new_text, classifier, vectorizer)
print(f"The predicted sentiment for the new text is: {predicted_sentiment[0]}")

if predicted_sentiment == 1:
    print('this is cyberbullying')
if predicted_sentiment == 0:
    print('this is not cyberbullying')


