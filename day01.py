'''Q1) function to reverse a string'''
def reverse_string(s): # Use slicing to reverse the string
    return s[::-1]
print(reverse_string('Hello World!'))
# Output: !dlroW olleH

'''Q2) check if string is palindrome'''
def is_plaindrome(s): # Compare the string with its reverse
    return s == s[::-1]
print(is_plaindrome('madam'))
# Output: True

'''Q3) function to find the nth Fibonacci number using recursion'''
def fibonacci(n): # Base case: if n is 0 or 1, return n (0,1,1,2,3,5,8,13...)
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2) # Recursive case: sum of the previous two numbers
    
n = 5
print(f"The {n}th Fibonacci number is : {fibonacci(n)}")

'''Q4) indices of two numbers that add up to a specific target in an array'''
# Brute force approach
def two_sum(nums, target):
    n = len(nums)
    for i in range(n): 
        for j in range(i+1, n):
            if nums[i] + nums[j] == target:
                return [i,j]

print(two_sum([2, 7, 3, 15], 10))  
# Output: [1, 2]

# Optimized approach
def two_sum(nums, target):
    num_map = {} # Start with an empty dictionary
    for index, num in enumerate(nums):  # enumerate() function returns both the index and the value of the element in the list
        complement = target - num
        if complement in num_map:
            return [num_map[complement], index] # Return the indices of the two numbers
        num_map[num] = index # Store the number and its index in the dictionary

print(two_sum([2, 7, 3, 15], 10))  
# Output: [1, 2]

'''Q5) function to calculate the factorial of a number'''
def factorial(n):   # Base case: if n is 0 or 1, return 1 (0! = 1, 1! = 1, 2! = 2, 3! = 6(3x2x1), 4! = 24(4x3x2x1), 5! = 120(5x4x3x2x1))
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n-1) # Recursive case: multiply n by the factorial of n-1

print(factorial(5))
# Output: 120
    