#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv

# Read the CSV file and create a list of rows
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


# In[2]:


#Make unique values list and be so so happy 
unique_values = set()

for row in data[1:]:
    unique_values.add(row[1])

unique_values_list = list(unique_values)

print(unique_values_list)


# In[3]:


#add to normal data
data.append(unique_values_list)

for row in data:
    print(row)


# In[6]:


# Define a custom key function for sorting
def sort_key(row):
    return row[1].upper()

# Sort the rows from 1 and leave out the first and last row
sorted_data = [data[0]] + sorted(data[1:-1], key=sort_key) + [data[-1]]

for row in sorted_data:
    print(row)


# In[7]:


#make csv file
output_file = 'P:\Desktop\Thesis data\sorted_data.csv'

with open(output_file, 'w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows(sorted_data)


# In[ ]:





# In[ ]:





# In[ ]:




