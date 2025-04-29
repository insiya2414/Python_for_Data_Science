'''Q6) count the occurrences of each element in a list'''
#using a dictionary
def count_occurences(list):
    count={} #create a dictionary to store the count of each element
    for i in list: #iterate through the list
        if i in count: #if the element is already in the dictionary, increment the count
            count[i]+=1
        else: #if the element is not in the dictionary, add it to the dictionary with a count of 1
            count[i]=1
    return count

list=[1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10]
print(count_occurences(list))
# Output: {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2}

#using collections.Counter
from collections import Counter
def count_occurrences(list):
    return Counter(list)

print(count_occurrences([1, 2, 2, 3, 3, 3]))
# Output: Counter({3: 3, 2: 2, 1: 1})

'''Q7) load a CSV file into a Pandas DataFrame'''
import pandas as pd #import the pandas library
df = pd.read_csv('data.csv') #load the csv file into a dataframe

print(df.head()) #print the first 5 rows of the dataframe
print(df.info()) #print information about the dataframe
print(df.describe()) #print descriptive statistics of the dataframe
print(df.shape) #print the number of rows and columns in the dataframe
print(df.columns) #print the column names of the dataframe
print(df.dtypes) #print the data types of the columns
print(df.isnull().sum()) #print the number of missing values in each column

'''Q8) function to calculate the element-wise sum of two Numpy arrays'''
import numpy as np
arr1 = np.array([1,2,3,4,5])
arr2 = np.array([6,7,8,9,10])

def element_wise_sum(arr1, arr2):
    return arr1 + arr2
print(element_wise_sum(arr1, arr2))
# Output: [ 7  9 11 13 15]

# OR

import numpy as np
arr1 = np.array([1, 2])
arr2 = np.array([4, 5])
result = np.add(arr1, arr2) #add the two arrays element-wise
print(result)
# Output: [5 7]

'''Q9) extract the diagonal elements of a Numpy matrix '''
import numpy as np
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(np.diagonal(matrix)) #print the diagonal elements of the matrix with the help of a function called 'np.diagonal'.
#[1 2 3]
#[4 5 6]
#[7 8 9]
# Output: [1 5 9]

'''Q10)  reshape a 1D Numpy array into a 2D array with 3 rows'''
import numpy as np
arr = np.array([1,2,3,4,5,6])
reshaped_arr = np.reshape(3,2) #reshape the array into 3 rows and 2 columns
print(reshaped_arr)
# Output: [[1 2]
#          [3 4]
#          [5 6]]






