'''Q41) types of EDA'''
# 1. Descriptive Statistics:
# Mean, median, mode, standard deviation, range, quartiles, etc.
# 2. Data Visualization:
# Histograms, scatter plots, box plots, bar charts, etc.
# 3. Correlation and Relationships:
# Correlation matrix, scatter plot matrix, etc. 
# 4. Outliers and Anomalies:
# Box plots, scatter plots, etc.
# 5. Data Distribution:
# Histograms, box plots, etc.
# 6. Time Series Analysis:
# Trend analysis, seasonality, etc.     
# 7. Missing Values:
# Missing value patterns, imputation methods, etc.
# 8. Data Transformation:
# Normalization, standardization, etc.
# 9. Feature Engineering:
# Creating new features from existing ones. 
# 10. Hypothesis Testing:
# Testing statistical hypotheses about the data.    


'''Q42) most common n-grams in a given text dataset'''
from nltk import ngrams          # Import n-gram generator from nltk
from collections import Counter  # Import Counter to count frequency

def most_common_ngrams(text, n, top_k): #text: the input text string., n: the size of n-grams (e.g., 2 for bigrams, 3 for trigrams)., top_k: how many of the most common n-grams to return.
        words = text.split() # Splits the text into a list of words:
        n_grams = list(ngrams(words, n)) # Creates a list of n-word sequences.
        return Counter(n_grams).most_common(top_k) # Uses Counter to count frequency of each n-gram.

text = "data science is fun and data science is interesting"
print(most_common_ngrams(text, 2, 2))
[(('data', 'science'), 2), (('science', 'is'), 2)]
# Output : [('data', 'science'), ('science', 'is')]
# Meaning: ('data', 'science') and ('science', 'is') each appear twice.


'''Q43) a pivot table from raw transactional data'''
# a pivot table, which organizes the data into a table format with aggregated values.
# A pivot table allows you to:
# Group your data by one or more keys (like "Category")
# Summarize data using functions like sum, mean, count, etc.
# Reshape the data into a readable, matrix-like format

import pandas as pd

def create_pivot_table(df, index, columns, values, aggfunc):
    return pd.pivot_table(df, index=index, columns=columns, values=values, aggfunc=aggfunc)

# What Each Parameter Means:
# df: Your DataFrame
# index: Row categories (e.g., 'Category')
# columns: Column categories (e.g., 'Type')
# values: The data to aggregate (e.g., 'Value')
# aggfunc: Aggregation function like 'sum', 'mean', 'count', etc.

data = {'Category': ['A', 'A', 'B'], 'Type': ['X', 'Y', 'X'], 'Value': [10, 20, 30]}
df = pd.DataFrame(data)


pivot_table = create_pivot_table(df, index='Category', columns='Type', values='Value', aggfunc='sum')
print(pivot_table)
# Output: Type         X     Y
        # Category            
        # A         10.0  20.0
        # B         30.0   NaN

'''Q44) Given an integer array, find the sum of the largest contiguous subarray within the array. 
For example, given the array A = [0,-1,-5,-2,3,14] it should return 17 because of [3,14]. 
Note that if all the elements are negative it should return zero'''
def max_subarray_sum(arr):
      n = len(arr)
      max_sum = arr[0]
      curr_sum = 0
      for i in range (n):
            curr_sum = curr_sum + arr[i]
            if curr_sum > max_sum:
                  max_sum = curr_sum
            if curr_sum < 0:
                  curr_sum = 0
      return max_sum

A = [0,-1,-5,-2,3,14]
print(max_subarray_sum(A))
# Output: 17

'''Q45) Extract entities (e.g., person names, locations) from a given text using Python libraries'''
# Extracting named entities (like people, places, organizations) from text using spaCy, a powerful NLP (Natural Language Processing) library in Python.
# What is Named Entity Recognition (NER)?
# NER is a process in NLP where the system identifies "named entities" in text. These entities can be:
# 🧑 PERSON – Names of people (e.g., Barack Obama)
# 🗺️ GPE – Countries, cities, states (e.g., Hawaii)
# 🏢 ORG – Organizations (e.g., Google)
# 📅 DATE – Dates
# 💰 MONEY – Monetary values, etc.


import spacy

def extract_entities(text):
    # Load the English NLP model (small size)
    nlp = spacy.load("en_core_web_sm")

    # Process the input text using the model
    doc = nlp(text)

    # Extract entity text and its label (e.g., PERSON, GPE)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Test the function
print(extract_entities("Barack Obama was born in Hawaii."))
# Output: [('Barack Obama', 'PERSON'), ('Hawaii', 'GPE')]
