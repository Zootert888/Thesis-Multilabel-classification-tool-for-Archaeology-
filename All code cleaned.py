#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# All libaries
import csv
import os
import pandas as pd
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


# Replace all unknown values from the original csv file with an empty string
# Read the CSV file and create a list of rows
# Original file with all data = oasis_period_20230822-111002.csv 
data = []
with open('P:\Desktop\Thesis data\oasis_period_20230822-111002.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data.append(row)

# List of values to identify and replace
values_to_replace = ['uncertain', 'no period', 'none', 'period unknown', 'period unassigned']

# Clean the data by replacing values with an empty string
for row in data:
    for i, value in enumerate(row):
        if value in values_to_replace:
            row[i] = ''

#print cleaned data
for row in data:
    print(row)


# In[ ]:


# Make unique values list 
# Where every unique value from csv file is a period mentioned 
unique_values = set()

for row in data[1:]:
    unique_values.add(row[1])

unique_values_list = list(unique_values)

# Add data to csv file
data.append(unique_values_list)

for row in data:
    print(row)

print(unique_values_list)
# Print to check if there are no doubles due to different spellings


# In[ ]:


# Sort the csv file and make leave out the header and unique values list
# Define a custom key function for sorting
def sort_key(row):
    return row[1].upper()

# Sort the rows from 1 and leave out the first (header) and last row (unique value list)
sorted_data = [data[0]] + sorted(data[1:-1], key=sort_key) + [data[-1]]

for row in sorted_data:
    print(row)


# In[ ]:


# Make csv file
output_file = 'P:\Desktop\Thesis data\sorted_data.csv'

with open(output_file, 'w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows(sorted_data)


# In[ ]:


# Delete all oasis ids with no labelled periods from csv, and make a new csv
# This might seem redundant and could've been done in above code but this was made clear to me after the fact
# This however is the most accurate of representation of how i worked

# Define the CSV file path and new storing data
input_file_path = 'P:/Desktop/Thesis data/sorted_data.csv'
output_file_path = 'P:/Desktop/Thesis data/final.csv'

# Open the input and output CSV files
with open(input_file_path, 'r', newline='') as input_file, open(output_file_path, 'w', newline='') as output_file:
    csv_reader = csv.reader(input_file)
    csv_writer = csv.writer(output_file)

     # Skip the first row
    next(csv_reader)

    first_row = next(csv_reader)
    csv_writer.writerow(first_row)
    
    for row in csv_reader:
        # Check if all values from the second to the thirteenth are empty strings
        if all(value == '' for value in row[1:13]):
            continue  # Skip this row
        else:
            csv_writer.writerow(row)  
# This created the final.csv file which had no empty strings so only labelled as periods


# In[ ]:


# Connect the csv labels to the text files i have saved under ADS-dump-UTF8
# This is a test to check if files correspond

# Paths to CSV file and directory containing text files
csv_file = r'P:\Desktop\Thesis data\final_data.csv'
text_files_path = r'ADS-dump-UTF8\\'

# Read CSV file
with open(csv_file, 'r', newline='', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    header = next(csv_reader)  # Header skiped

    for row in csv_reader:
        document_filename = row[0]
        expected_filename = text_files_path + document_filename + '_1.txt'
        print(expected_filename)
        
        if os.path.exists(expected_filename):
            print(f"File connected for {document_filename}")
        else:
            print(f"Text file not found for {document_filename}")


# In[ ]:


# Add new column to later add first 1000 words of text files cleaned up and make new csv file 

new_column_header = "Text files"

# Open the existing CSV file for reading
with open('P:\\Desktop\\Thesis data\\final.csv', mode='r') as read_file:
    csv_reader = csv.reader(read_file)
    
    # Read the existing rows
    rows = list(csv_reader)
    
    # Add the new column header to the first row
    rows[0].append(new_column_header)
    
    # Add an empty value for laterfilled in text
    for i in range(1, len(rows)):
        rows[i].append("")
    
# Open the CSV file in write mode to update with the new empty column
with open('P:\\Desktop\\Thesis data\\final_data.csv', mode='w', newline='') as write_file:
    csv_writer = csv.writer(write_file)
    
    # Write the updated rows back to the file
    csv_writer.writerows(rows)

print("New empty column added to the CSV file.")


# In[ ]:


# Clean up text from files and add to dataframe/csv file -> master.csv/master_df
# Define the header columns
header_columns = [
    'oasis ids', 'early medieval', 'late mesolithic', 'medieval', 'post medieval',
    'later prehistoric', 'early iron age', 'middle palaeolithic', 'neolithic',
    'late iron age', 'bronze age', 'early bronze age', 'late prehistoric', 'roman',
    'middle iron age', 'late neolithic', 'early neolithic', 'middle bronze age',
    'early mesolithic', 'lower palaeolithic', 'upper palaeolithic', 'late bronze age',
    'palaeolithic', 'early prehistoric', 'mesolithic', '20th century',
    'middle neolithic', 'iron age', 'nil antiquity', 'Text files'
]

# Create a new empty dataframe
master_df = pd.DataFrame(columns=header_columns)

# Define paths
text_files_directory = r'ADS-dump-UTF8\\'

# Open the CSV file for reading
csv_file_path = r'P:\Desktop\Thesis data\final.csv'
with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    # Get the header row to identify columns
    header_row = next(csv_reader)
    
    # Iterate through the rows in the CSV final file
    for row in csv_reader:
        oasis_id = row[0]

        # Check if the corresponding text file exists
        text_file_path = os.path.join(text_files_directory, oasis_id + '_1.txt')
        if os.path.exists(text_file_path):
            # Read the text content
            with open(text_file_path, 'r', encoding='utf-8') as text_file:
                text_content = text_file.read()
                
                # Replace line endings with spaces instead of new line
                text_content = text_content.replace('\r', ' ').replace('\n', ' ')
               
                #Take only first 1000 words to append to master.csv file
                text_first_1000_words = ' '.join(text_content.split()[:1000])
                
                # Download stopwords file from nltk library
                nltk.download('stopwords') 
               
                # Tokenize the 1000 words text
                from nltk.tokenize import word_tokenize
                tokenized_text = word_tokenize(text_first_1000_words)
                
                # Make stopwords list
                from nltk.corpus import stopwords
                stopwords = stopwords.words('english')

                # Make punctuation list
                from string import punctuation
                punctuation = list(punctuation)

                # Remove stopwords and punctuation from 1000 words text
                cleaned_tokens = [token for token in tokenized_text if token not in stopwords and token not in punctuation]

                # Change tokens list back to string and make string lowercase
                cleaned_text = ''.join(cleaned_tokens).lower()
                
                # Example of what the rows in the new file should look like 
                new_row = [oasis_id, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, cleaned_text]
                # Append the new row to the master_df
                master_df.loc[len(master_df)] = new_row
                
                # Check if any of the periods are in columns 1 to 14
                periods = [
                    'early medieval', 'late mesolithic', 'medieval', 'post medieval',
                    'later prehistoric', 'early iron age', 'middle palaeolithic', 'neolithic',
                    'late iron age', 'bronze age', 'early bronze age', 'late prehistoric', 'roman',
                    'middle iron age', 'late neolithic', 'early neolithic', 'middle bronze age',
                    'early mesolithic', 'lower palaeolithic', 'upper palaeolithic', 'late bronze age',
                    'palaeolithic', 'early prehistoric', 'mesolithic', '20th century',
                    'middle neolithic', 'iron age', 'nil antiquity'
                ]
    
                for period in periods:
                    if period in row[1:14]: 
                        # If the period is in the column of final_data.csv, add 1 to the corresponding column in master_df
                        master_df.loc[len(master_df) - 1, period] = 1

# Save the resulting master_df to a CSV file for easier viewing 
master_df.to_csv(r'P:\Desktop\Thesis data\master.csv', index=False)
print("master.csv has been saved.")


# In[ ]:


# Kdnugget tutorial regarding Multilabel classification tool 
# Load CSV master file
csv_file = 'P:\\Desktop\\Thesis data\\master.csv'

# The first row in the CSV file contains the headers
df = pd.read_csv(csv_file)

# Replace NaN values in the 'Text files' column with an empty string
df['Text files'].fillna('', inplace=True)

X = df['Text files']

y = np.asarray(df[df.columns[1:29]])

# print to check if this is the correct size
print(len(y))

# Transform the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=30, max_df=0.9)
X_tfidf = vectorizer.fit_transform(X)

print(X_tfidf)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=101)

# Build multi-output classifier with Logistic Regression
clf = MultiOutputClassifier(LogisticRegression()).fit(X_train, y_train)

# Use clf to make predictions on your multi-label classification task
y_pred = clf.predict(X_test)


# In[ ]:


# Predictive multilabel classification model accuracy score
print('Accuracy Score: ', accuracy_score(y_test, y_pred))


# In[ ]:


# Calculate metrics for the classification report
periods= ['early medieval', 'late mesolithic', 'medieval', 'post medieval',
    'later prehistoric', 'early iron age', 'middle palaeolithic', 'neolithic',
    'late iron age', 'bronze age', 'early bronze age', 'late prehistoric', 'roman',
    'middle iron age', 'late neolithic', 'early neolithic', 'middle bronze age',
    'early mesolithic', 'lower palaeolithic', 'upper palaeolithic', 'late bronze age',
    'palaeolithic', 'early prehistoric', 'mesolithic', '20th century',
    'middle neolithic', 'iron age', 'nil antiquity']
print(classification_report(y_test, y_pred, target_names=periods))


# In[ ]:


# Give SVC Classification
clf = MultiOutputClassifier(SVC()).fit(X_train, y_train)
y_pred_svc = clf.predict(X_test)
print(classification_report(y_test, y_pred_svc, target_names=periods))


# In[ ]:


# Give RandomForestClassifier Classification
clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=10)).fit(X_train, y_train)
y_pred_rf = clf.predict(X_test)
print(classification_report(y_test, y_pred_rf, target_names=periods))


# In[ ]:


# Give DecisionTreeClassifier Classification
clf = MultiOutputClassifier(DecisionTreeClassifier()).fit(X_train, y_train)
y_pred_dt = clf.predict(X_test)
print(classification_report(y_test, y_pred_dt, target_names=periods))

