#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('ResumeDataSet.csv')

# Display the first few rows of the dataset
df.head()

# Check the distribution of resume categories
df['Category'].value_counts()

# Visualize the distribution of resume categories
plt.figure(figsize=[15, 10])
sns.countplot(df['Category'])
plt.xticks(rotation=90)
plt.show()

# Calculate and visualize the category distribution using a pie chart
counts = df['Category'].value_counts()
labels = df['Category'].unique()

# Define colors for the pie chart
colors = ['blue', 'red', 'green', 'orange', 'purple', 'pink', 'cyan', 'yellow']

plt.figure(figsize=(15, 10))
plt.pie(counts, labels=labels, autopct="%1.1f%%", shadow=True, colors=colors)
plt.show()

# Define a function to clean the text in resumes
def cleanResume(txt):
    cleantxt = re.sub('http\S+\s', ' ', txt)
    cleantxt = re.sub('RT|cc', ' ', cleantxt)
    cleantxt = re.sub('#\S+\s', ' ', cleantxt)
    cleantxt = re.sub('@\S+', '  ', cleantxt)  
    cleantxt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleantxt)
    cleantxt = re.sub(r'[^\x00-\x7f]', ' ', cleantxt) 
    cleantxt = re.sub('\s+', ' ', cleantxt)
    return cleantxt

# Apply the cleanResume function to the 'Resume' column
df['Resume'] = df['Resume'].apply(lambda x: cleanResume(x))

# Encode the 'Category' column using LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df['Category'])
df['Category'] = le.transform(df['Category'])

# Create a TfidfVectorizer for text data
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Fit the TfidfVectorizer on resume text
tfidf.fit(df['Resume'])

# Transform the resume text data into TF-IDF features
requiredText = tfidf.transform(df['Resume'])

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(requiredText, df['Category'], test_size=0.2, random_state=42)

# Build a K-Nearest Neighbors (KNN) classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)
ypred = clf.predict(X_test)

# Calculate and print the accuracy of the KNN classifier
print("Accuracy Score:", accuracy_score(y_test, ypred))

# Export the trained TF-IDF vectorizer and classifier to pickle files
import pickle
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
pickle.dump(clf, open('clf.pkl', 'wb'))

# Define a sample resume
myresume = """Greater Noida, India
+91 8........3
naveenpandey2706@gmail.com
CONTACT
SKILLS
EDUCATION
PROFESSIONAL EXPERIENCE
... (Your resume text) ...
TO OBTAIN A CHALLENGING POSITION AS A DATA SCIENTIST, WHERE I CAN APPLY MY SKILLS AND KNOWLEDGE TO DRIVE BUSINESS GROWTH AND SUCCESS."""

# Load the trained classifier and TF-IDF vectorizer
clf = pickle.load(open('clf.pkl', 'rb'))
input_features = tfidf.transform([cleanResume(myresume)])

# Make a category prediction for the input resume
prediction_id = clf.predict(input_features)[0]

# Map category ID to category name
category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    # ... (Add more category mappings as needed)
}

category_name = category_mapping.get(prediction_id, "Unknown")

# Print the predicted category name
print("Predicted Category:", category_name)
print(prediction_id)
