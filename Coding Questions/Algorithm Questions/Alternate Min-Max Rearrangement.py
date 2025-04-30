'''Modify a given array of integers so that the first element is the smallest, 
the second is the largest, the third is the second-smallest, the fourth is the 
second-largest, and so on.'''

def modify_array(arr):
    arr.sort() #sort the array
    left = 0 #initialize the left pointer: starts at the beginning (smallest element).
    right = len(arr)-1 #initialize the right pointer: starts at the end (largest element).
    result = [] #initialize an empty list to store the result
    
    while left <= right: #while the left pointer is less than or equal to the right pointer
        result.append(arr[left]) #append the smallest element to the result
        left +=1 #increment the left pointer
        if left <= right: #if the left pointer is still less than or equal to the right pointer
            result.append(arr[right]) #append the largest element to the result
            right-=1 #decrement the right pointer
            
    return result

arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(modify_array(arr))
# Output: [1, 10, 2, 9, 3, 8, 4, 7, 5, 6]

