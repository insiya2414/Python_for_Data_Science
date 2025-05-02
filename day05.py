'''Q26 transpose a NumPy array'''
import numpy as np 
array = ([1,2],[4,5])
transposed_array = array.T # Transpose the array using the .T attribute - swaps rows and columns
print(transposed_array)
# Output: [[1  4]
#          [2  5]]

'''Q27) median of two sorted arrays of different sizes'''
def median(arr1, arr2):
    # Merge and sort both arrays, then find median:
    # If length is even, average the two middle numbers
    # If length is odd, take the middle number
    merged = sorted(arr1+arr2)
    n = len(merged)
    if n % 2 == 0:
        return (merged[n // 2] + merged[n // 2 - 1]) / 2
    else:
        return merged[n // 2]

print(median([1, 3], [2, 4, 5]))
# Output : 3

'''Q28) Implement a sliding window to find the maximum sum of a subarray of a given size k'''
def max_sum_subarray(arr, k):
    window_sum = sum(arr[:k])      # sum of the first window of size k
    max_sum = window_sum           # initialize max sum
    for i in range(k, len(arr)):   # slide the window
        window_sum += arr[i] - arr[i - k]  # add new, remove old
        max_sum = max(max_sum, window_sum)  # update max if needed
    return max_sum
print(max_sum_subarray([2, 1, 5, 1, 3, 2], 3))
# Output: 3

'''Q29) Find the kth smallest element in an unsorted array'''
import heapq # For large dataset use min-heap(prority queue)
def kth_smallest(arr,k):
    return heapq.nsmallest(k, arr)[-1] # Returns the kth smallest element by getting k smallest elements and taking the last one

print(kth_smallest([7, 10, 4, 3, 20, 15], 3))
# Output: 7
# OR
def kth_smallest(arr, k):
    arr.sort()
    return arr[k - 1]  # because list is 0-indexed

'''Q30) Generate all possible permutations of a given list of numbers'''
from itertools import permutations # Import permutations from itertools module to generate all possible arrangements of elements

def generate_permutations(arr):
    return list(permutations(arr))

print(generate_permutations([1, 2, 3]))
# Output: [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]