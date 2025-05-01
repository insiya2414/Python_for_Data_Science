'''Given a number as an input, find the following smallest palindrome number. 
For simplicity, assume the input value will not exceed 1 million. 
The next palindrome may be greater than 1 million.'''

# A palindrome is a number (or string) that reads the same forward and backward.
# Example: 121, 1331, 12321, etc.

def is_palindrome(num):
    return str(num) == str(num)[::-1] # Convert number to string and compare with its reverse

def next_smallest_palindrome(n):
    n += 1  # Start checking from the next number
    while True: 
        if is_palindrome(n):
            return n
        n += 1

# Test Cases
print(next_smallest_palindrome(123))      # Output: 131
print(next_smallest_palindrome(999999))   # Output: 1000001
print(next_smallest_palindrome(123456))   # Output: 124421
