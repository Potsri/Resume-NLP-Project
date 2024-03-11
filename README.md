# Resume-NLP-Project
The NLP Resume Classification project involves building a text classification model to predict the job profile category for a given resume. The goal is to help HR professionals automate the process of resume screening and candidate selection. The project uses various machine learning algorithms and NLP techniques for preprocessing the text data and evaluating the performance of the model.

# Business Objective:

The objective of this project is to classify resumes into different categories based on the job profile. The project involves a text classification problem where the goal is to predict the job profile category for a given resume. The dataset contains resumes for various job profiles and their respective categories. The project aims to help HR professionals automate the process of resume screening and candidate selection.

# Dataset: 
The dataset contains resumes for various job profiles and their respective categories. The dataset includes the following variables:

# Resume Text: 
The text data of the resumes. Category: The job profile category for the given resume.

# Steps Involved In These Project

- Installation
  
  To run this project, you will need to have Python 3 installed along with the following libraries:

- Pandas
- Numpy
- Scikit-learn
- NLTK
- Docx2txt
- Streamlit

Step 1 : Extracted The Data From Resumes Using Some Libraries Docx2txt, Texttract, Doc, Pyrespar, OS, RE , And We Have Created A Data Set .

Step 2 : Eda( Exploratory Data Analysis) * Data Preprocesssing * Data Understanding

Step 3 : Visualizations

Step 4 : Model Building With Different Algorithms

Step 5 : Deployment

Step 6 : Production.

# Approach: 
The project involves building a text classification model to predict the job profile category for a given resume. Various machine learning algorithms such as Naive Bayes, Support Vector Machines, and Random Forest can be used for this purpose. Natural Language Processing (NLP) techniques such as tokenization, stemming, and lemmatization can be used to preprocess the text data. The performance of the model can be evaluated using metrics such as accuracy, precision, recall, and F1-score.

A dictionary or table which has various skill sets categorised is maintained. If we have words like keras, tensorflow, CNN, RNN appearing in the candidate CV, then they are merged under one column titled ‘Deep Learning’.

We have used NLP algorithm that parses the whole resume and searches for the words mentioned in the dictionary or table.

The next step is to count the occurrence of the words under various category for each candidate.

We represent the above information in a visualized manner so that it becomes easier for the recruiter to choose the candidate. At the same time a csv file is also created which gives a score card of the different skills acquired by each candidate.

# Data Preparation
The data for this project consisted of resumes in doc and docx formats. The first step was to read the resumes using the docx2txt library and convert them into a dataframe. The data was then preprocessed using NLTK to remove stop words, punctuation, and perform stemming.

The preprocessed data was then visualized using various plots to gain insights into the characteristics of the data.

# Model Training
Various classification algorithms were trained on the preprocessed data, including K-Nearest Neighbors (KNN), Naive Bayes, Random Forest, and Support Vector Machines (SVM). The performance of each algorithm was evaluated using various metrics such as accuracy, precision, recall, and F1 score.

SVM was chosen as the main model due to its superior performance in the project.

# Deployment
The final model was deployed on Streamlit, which is a platform for building and sharing data applications. The outcome of the project was a model that could take a resume as input and output which job profile the resume matched, along with the candidate's mobile number and email address, and the skills listed on the resume.

# Conclusion
The project demonstrates the effectiveness of NLP techniques for analyzing and processing textual data. The model can automate and improve the efficiency of tasks such as resume screening and job matching. The project has the potential to be extended to other languages and to incorporate more advanced NLP techniques.

# Conclusion:
The key challenges in this project include handling the variability in resume formats, extracting meaningful features from unstructured text data, and building a robust classification model that can generalize well to unseen resumes. The system should be scalable, adaptable to different industries and job markets, and capable of handling a diverse range of resumes.

The success of this project will be measured by the accuracy and efficiency of the resume classification system. The model should achieve high classification accuracy on a representative dataset of resumes and demonstrate the ability to handle new resumes with consistent performance.

Classifying resumes into different job profile categories can help HR professionals automate the process of resume screening and candidate selection. By using the given text data, a classification model can be built to predict the job profile category for a given resume. The model's performance can be evaluated using various metrics, and the best model can be selected for deployment.



