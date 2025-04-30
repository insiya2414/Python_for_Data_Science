'''Given a 2D array of user IDs, find the number of friends a user has. 
Note that users can have none or multiple friends.'''

def count_friends(user_ids):
    friend_count= {} #create a dictionary to store the number of friends each user has
    for row in user_ids: #iterate through each row in the 2D array
        for user_id in row: #iterate through each user_id in the row
            if user_id not in friend_count: #if the user_id is not in the dictionary, add it and set the count to 0
                friend_count[user_id] = 0
            friend_count[user_id] +=1 #increment the count for the user_id
    return friend_count

user_ids = [[1,2,3], [4,5,6], [7,8,9]] # List of lists in Python
print(count_friends(user_ids))
# Output: {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}

# using collections.Counter
from collections import Counter

def count_friends(user_ids):
    return Counter(user_id for row in user_ids for user_id in row) # Counter is a dictionary subclass that counts hashable objects


