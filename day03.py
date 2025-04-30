'''Q11) function to calculate the mean, median, and standard deviation of a list of numbers'''
import numpy as np
list = [1,2,3,4,5,6,7,8,9,10]
mean = np.mean(list)
median = np.median(list)
std_dev = np.std(list)
print(mean,median,std_dev)
# Output: 5.5 5.5 2.8722813232690143

# Mean: Average of all numbers
# → Add all values, divide by count

# Median: Middle value when sorted
# → If even count, average the two middle numbers

# Mode: Most frequent value
# → Can be more than one if there's a tie

# Standard Deviation: Spread of values around the mean
# → Low = values close to mean, High = values more spread out

'''Q12)  handle missing values in a DataFrame'''
# Drop rows with missing values: df.dropna()

# Drop columns with many missing values: df.dropna(axis=1)

# Fill with a value (e.g., 0 or mean): df.fillna(0) or df.fillna(df['col'].mean(), inplace = True)

# Forward fill (use previous value): df.fillna(method='ffill')

# Backward fill (use next value): df.fillna(method='bfill')

# Interpolate missing values: df.interpolate() [Interpolate means to estimate missing values by using the values before and after them]

# Check for missing values: df.isnull().sum()

'''Q13) read and write data from a file in Python'''
# Read data from a file:
with open('file.txt', 'r') as file: # use the built-in open() function
    content = file.read()
print(content)

# Write data to a file:
with open('file.txt', 'w') as file:
    file.write('Hello, world!')

'''Q14) SQL query to retrieve all columns from a table named employees where the age is greater than 30 '''
SELECT *
FROM employees
WHERE age > 30;

'''Q15) SQL query to join two tables: orders and customers, where the customer_id in orders matches the id in customers'''
SELECT *
FROM orders AS o
JOIN customers AS c
ON o.customer_id = c.id;

'''Q16) SQL query to find the average salary for each department in a company table, but only for departments with more than 10 employees'''
SELECT
department, AVG(salary) AS avg_salary
FROM company
GROUP BY department
HAVING COUNT(employee_id) > 10;

'''Q17) SQL query to find all employees whose salary is greater than the average salary in the employees table'''
SELECT *
FROM employees
HAVING salary > (SELECT AVG(salary) FROM employees);

'''Q18) SQL query to find the total sales from the sales table for each product'''
SELECT product_id, SUM(sales_amount) AS total_sales
FROM sales
GROUP BY product_id;

'''Q-Extra) function to check if a number is prime'''
def is_prime(num):
    if num <= 1:
        return False
    for i in range(2,int(num**0.5)+1): # int(num**0.5)+1 is the square root of the number
        if num % i == 0: # if the number is divisible by any number other than 1 and itself, it is not prime
            return False
    return True
print(is_prime(11))
# Output: True
# Example run:
# num = 11
# num**0.5 = 3.316 -> taking square root of 11
# int(num**0.5) = 3 -> taking the integer part of the square root
# range(2, int(num**0.5) + 1) → range(2, 4) -> creating a range from 2 to 3

