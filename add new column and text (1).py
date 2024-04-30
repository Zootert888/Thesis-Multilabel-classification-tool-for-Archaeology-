#!/usr/bin/env python
# coding: utf-8

# In[13]:


#add new column
import csv

new_column_header = "Text files"

# Open the existing CSV file for reading
with open('P:\\Desktop\\Thesis data\\final_data.csv', mode='r') as read_file:
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


# In[7]:


import csv
import os

# Directory containing text files
text_files_directory = r'Test\\'

# Open the CSV file for reading and writing
with open(r'P:\Desktop\Thesis data\final_data.csv', mode='r+', newline='', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    rows = list(csv_reader)
    
    # Get the header row to identify columns
    header_row = rows[0]
    
    # Get the text content from the text files
    text_contents = []
    for text_file_name in os.listdir(text_files_directory):
        text_file_path = os.path.join(text_files_directory, text_file_name)
        with open(text_file_path, 'r', encoding='utf-8') as text_file:
            text_content = text_file.read()
            text_contents.append(text_content)
    
    # Update the specific row (index 14) with the text content
    rows[14][header_row.index('Text files')] = ' '.join(text_contents)
    
    # Write rows back to the CSV file
    csv_file.seek(0)
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(rows)

print("Text content added to the row 14 in the CSV file.")


# In[4]:


import pandas as pd
pd.read_csv(r'P:\Desktop\Thesis data\final_data.csv')


# In[ ]:




