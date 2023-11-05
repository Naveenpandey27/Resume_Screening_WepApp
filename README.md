# Resume_Screening_WepApp
I designed and built a user-friendly web application using Python and Streamlit. This app streamlines the resume screening process by efficiently analyzing and categorizing resumes. It simplifies the initial screening of job applicants, making the hiring process faster and more accurate.

# Resume Classification using Python

![resumesc](https://github.com/Naveenpandey27/Resume_Screening_WepApp/assets/66298494/4b51aab1-329a-438b-b902-4fe7deb49c4c)


This Python code is designed to classify resumes into different categories using the K-Nearest Neighbors (KNN) algorithm. The code processes the resume text, applies TF-IDF (Term Frequency-Inverse Document Frequency) vectorization, and trains a KNN classifier for the task. Additionally, it allows you to make category predictions for new resumes. This README provides a brief overview of the code and how to use it.

## Getting Started

To get started with this code, follow these steps:

1. **Prerequisites**: Ensure that you have Python installed on your system. You may need to install additional libraries if not already present. You can use `pip` to install the required libraries:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

2. **Dataset**: The code assumes you have a dataset in a CSV file named 'ResumeDataSet.csv'. You should replace this file with your dataset or modify the code to load your dataset.

3. **Running the Code**: Run the code in a Python environment. This can be done by running the script in a Python IDE, a Jupyter Notebook, or from the command line.

## Code Explanation

The code performs the following steps:

1. **Data Loading**: It loads the resume dataset from 'ResumeDataSet.csv' using pandas and displays the first few rows of the dataset.

2. **Data Visualization**: It visualizes the distribution of resume categories using bar plots and a pie chart.

3. **Text Preprocessing**: The 'cleanResume' function is defined to clean the text in resumes, removing URLs, special characters, and extra white spaces.

4. **Label Encoding**: The 'Category' column is encoded using LabelEncoder for classification.

5. **Text Vectorization**: The code uses TF-IDF vectorization to convert text data into numerical features. It fits the TF-IDF vectorizer on the resume text and transforms it.

6. **Training and Testing Split**: The data is split into training and testing sets for model evaluation.

7. **KNN Classifier**: A K-Nearest Neighbors (KNN) classifier is built using the OneVsRestClassifier to classify resumes into different categories.

8. **Model Evaluation**: The accuracy of the KNN classifier is calculated and printed.

9. **Model Export**: The trained TF-IDF vectorizer and classifier are saved to pickle files ('tfidf.pkl' and 'clf.pkl') for future use.

10. **Category Prediction**: You can use the saved classifier to make category predictions for new resumes. An example resume is provided for testing, and the predicted category is printed.

## Customization

- To use your dataset, replace 'ResumeDataSet.csv' with your data file and adjust the code accordingly.
- You can add more category mappings in the `category_mapping` dictionary for your specific use case.

