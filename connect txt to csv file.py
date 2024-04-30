#!/usr/bin/env python
# coding: utf-8

# In[17]:


import os
import csv

# Paths to CSV file and directory containing text files
csv_file = r'P:\Desktop\Thesis data\final_data.csv'
text_files_path = r'ADS-dump-UTF8\\'

# Read CSV file
with open(csv_file, 'r', newline='', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    header = next(csv_reader)  # header

    for row in csv_reader:
        document_filename = row[0]
        expected_filename = text_files_path + document_filename + '_1.txt'
        print(expected_filename)
        
        if os.path.exists(expected_filename):
            print(f"File connected for {document_filename}")
        else:
            print(f"Text file not found for {document_filename}")



# In[ ]:




