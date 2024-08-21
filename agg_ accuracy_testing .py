import os
import zipfile
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pickle

rowamount = 5000
while rowamount < 100000:
    # Defines paths for extracting dataset
    zip_file_path = 'A Comprehensive Dataset for Automated Cyberbullying Detection (2).zip'
    extracted_dir = 'extracted_data'

    # Extract the ZIP file to the specified directory
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_dir)

    # Verify the contents of the extracted directory
    extracted_contents = os.listdir(extracted_dir)

    # Assuming the extracted directory contains a subdirectory
    subdir = extracted_contents[0]  # This would be 'A Comprehensive Dataset for Automated Cyberbullying Detection (2)'
    csv_file_name = '5. Communication_Data_Among_Users.csv'
    csv_file_path = os.path.join(extracted_dir, subdir, csv_file_name)

    # Read the CSV file
    df = pd.read_csv(csv_file_path, nrows=rowamount, delimiter=',', engine='python', quoting=3, skiprows=1, usecols=[4, 5], names=['review', 'status'])

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

    # Ensure the status column contains valid integer values
    df['status'] = df['status'].astype(int)

    # Vectorize the text data using CountVectorizer with binary=True and limited features
    vectorizer = CountVectorizer(binary=True, max_features=10000)  # Limit to top 10,000 features
    X = vectorizer.fit_transform(df['review'])  # Do not convert to array
    y = df['status']

    # Split the data into training and testing sets
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y, test_size=0.20, random_state=101)

    # Train and evaluate Logistic Regression separately
    classifier = LogisticRegression()
    classifier.fit(X_train_s, y_train_s)
    predictions = classifier.predict(X_test_s)
    conf_matrix = confusion_matrix(y_test_s, predictions)
    print(conf_matrix)
    print(classification_report(y_test_s, predictions))
    
    # Calculate and print the accuracy score with more precision
    accuracy = accuracy_score(y_test_s, predictions)
    print(f'Accuracy: {accuracy:.10f}')

    print(rowamount)
    rowamount = rowamount + 5000

print("process complete - yipee!!!")
