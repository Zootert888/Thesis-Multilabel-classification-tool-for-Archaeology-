#!/usr/bin/env python
# coding: utf-8

# In[23]:


#add new column
import csv

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


# In[8]:


import csv
import os
import pandas as pd

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
    
    # Iterate through the rows in the CSV file
    for row in csv_reader:
        oasis_id = row[0]

        # Check if the corresponding text file exists
        text_file_path = os.path.join(text_files_directory, oasis_id + '_1.txt')
        if os.path.exists(text_file_path):
            # Read the text content and extract the first 1000 words
            with open(text_file_path, 'r', encoding='utf-8') as text_file:
                text_content = text_file.read()
                # Replace line endings with spaces
                text_content = text_content.replace('\r', ' ').replace('\n', ' ')
                text_first_1000_words = ' '.join(text_content.split()[:1000])
                
                import nltk
                 # download stopwords file (only needs to run once!)
                #nltk.download('stopwords')
                
                # tokenize the text
                from nltk.tokenize import word_tokenize
                tokenized_text = word_tokenize(text_first_1000_words)


                # make stopwords list
                from nltk.corpus import stopwords
                stopwords = stopwords.words('english')

                # make punctuation list
                from string import punctuation
                punctuation = list(punctuation)

                # remove stopwords and punctuation from text
                cleaned_tokens = [token for token in tokenized_text if token not in stopwords and token not in punctuation]

                # change tokens list back to string
                cleaned_text = ''.join(cleaned_tokens).lower()
                
                new_row = [oasis_id, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, cleaned_text]
                # Append the new row to the master_df
                master_df.loc[len(master_df)] = new_row
                
                 # Check if any of the specified conditions are in columns 1 to 13
                periods = [
                    'early medieval', 'late mesolithic', 'medieval', 'post medieval',
                    'later prehistoric', 'early iron age', 'middle palaeolithic', 'neolithic',
                    'late iron age', 'bronze age', 'early bronze age', 'late prehistoric', 'roman',
                    'middle iron age', 'late neolithic', 'early neolithic', 'middle bronze age',
                    'early mesolithic', 'lower palaeolithic', 'upper palaeolithic', 'late bronze age',
                    'palaeolithic', 'early prehistoric', 'mesolithic', '20th century',
                    'middle neolithic', 'iron age', 'nil antiquity'
                ]
                # Check if any of the periods are in columns 1 to 13
                for period in periods:
                    if period in row[1:14]:  # Adjusted to include up to the 13th column
                        # If the condition is met, add 1 to the corresponding column in master_df
                        master_df.loc[len(master_df) - 1, period] = 1

# Save the resulting master_df to a CSV file
master_df.to_csv(r'P:\Desktop\Thesis data\master.csv', index=False)
print("master.csv has been saved.")


# In[1]:


# As requested by Amira
print("hello world")


# In[ ]:




