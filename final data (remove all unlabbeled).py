#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv

# Define the CSV file path and new store
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


# In[ ]:





# In[ ]:




