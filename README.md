README

Project description

This repository contains all the Pyhton scripts to process the documents within the CSV file. It cleans the data, analyzes the text files, and performs multilabel classification. The script reads a CSV file containing period labels, cleans the data, processes associated text files, and uses several classifiers to predict historical periods based on the text content.

This is done as an extension of the AGNES project created by Brandsen & Koole, 2022. To recreate multilabel classification on English archaeological documents. 

How to Install and Run the Project.

Before running the script, make sure you have the following libraries installed:
- csv
- os
- pandas
- nltk
- numpy
- scikit-learn

You can install the libraries using pip.

What the Project does.

- `all data cleaned.py` or `all data cleaned.ipynb`: Contains all the code necessary for this project.
- all the other .py/.ipynb files are  separated because of easier reference for Thesis purposes.

Files

- `oasis_period_20230822-111002.csv`: The original CSV file containing historical period data.
- `ADS-dump-UTF8\\`: Directory containing text files associated with the period data.
- `sorted_data.csv`:  File with sorted and cleaned data.
- `final.csv`: Intermediate file with rows containing period labels.
- `master.csv`: Final CSV file with cleaned text data and period labels.
- `Thesis data/`: Directory containing the CSV files and processed data.

 1. Data Cleaning
- Replace unknown values in the original CSV file with an empty string.
- Identify and clean specific values in the data.

 2. Unique Values Extraction

- Extract unique period labels from the data.
- Append the list of the unique values to the CSV data.

 3. Data Sorting

- Sort the CSV data by period labels.
- Write the sorted data to `sorted_data.csv`.

4. Data Filtering

- Remove rows without period labels.
- Create a new CSV file `final.csv` containing only labeled periods.

5. Text File Association

- Associate CSV labels with text files stored in the `ADS-dump-UTF8` directory.
- Check if text files correspond to the document filenames in the CSV file.

6. Data Preparation for Machine Learning

- Add a new column to the CSV file for cleaned text data.
- Clean the text files by removing stopwords and punctuation.
- Tokenize and lowercase the text data.
- Create `master.csv` with the cleaned text data and period labels.

7. Multi-label Classification

- Load the `master.csv` file.
- Perform TF-IDF vectorization on the text data.
- Split the data into training and testing sets.
- Train and evaluate various classifiers:
  - Logistic Regression
  - Support Vector Classifier (SVC)
  - RandomForestClassifier
  - DecisionTreeClassifier
- Generate classification reports for each classifier.

 How to use the Project
1. Make sure you have the necessary CSV and text files in the correct directories.
2. Run the script using Python.
3. The script will generate more CSV files and the final classification results.

Credit (apa 7)
Brandsen, A., & Koole, M. (2022). Labelling the past: Data set creation and multi-label classification of Dutch archaeological excavation reports. Language Resources and Evaluation, 56(2), 543–572. https://doi.org/10.1007/s10579-021-09552-6

Wijaya, C. Y. (2023). Multilabel classification: An introduction with Python's Scikit-Learn. KDnuggets. Retrieved from https://www.kdnuggets.com/multilabel-classification-an-introduction-with-pythons-scikit-learn

 License 
This project is licensed under the GNU General Public License v3.0. See the LICENSE file for details.


