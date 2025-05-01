'''Q19) return JSON data in a Flask route'''
from flask import Flask, jsonify #importing flask and jsonify

app = Flask(__name__) #creating a Flask app

@app.route('/data') #creating a route
def data():
    return jsonify({"name": "John", "age": 30}) # A Python dictionary being returned as JSON data

if __name__ == '__main__':
    app.run() #running the app  

'''Q20) create a simple “Hello, World!” app in Flask'''
from flask import Flask

app = Flask(__name__) #creating a Flask app
@app.route('/') #creating a route (root url)
def hello_world():
    return "Hello World!" #returning a string

if __name__ == '__main__':
    app.run() #running the app

'''Q21) explain how Flask handles HTTP methods like GET and POST'''
# | Method   | Purpose                              |
# |----------|--------------------------------------|
# | `GET`    | Fetch data from the server (default) |
# | `POST`   | Send data to the server              |
# | `PUT`    | Update existing data                 |
# | `DELETE` | Remove data from the server          |

# Flask allows you to specify which HTTP methods a route should respond to by using the methods parameter in 
# the @app.route() decorator. By default, Flask routes respond to GET requests, but you can specify others 
# such as POST, PUT, DELETE, etc.

@app.route('/form', methods=['GET', 'POST'])
def handle_form():
    if request.method == 'POST': # If the route receives a POST request (usually from a form submission), it returns "Form submitted!" 
        return 'Form submitted!'
    return 'Form not yet submitted. 
# If it’s a GET request (user just visits the page), it returns "Form not yet submitted."

# Flask is a web framework written in Python.

# It helps Python developers build web applications more easily.

# You write your code in Python, and Flask provides tools to manage routes, forms, databases, and more.

'''Q22) Class to Represent a Person with Basic Attributes'''
class Person:
    # Constructor to initialize the attributes
    def __init__(self, name, age, gender): # The __init__ method is used to initialize the object's attributes (i.e., set up the initial state of the object).
        self.name = name # self refers to the current object
        self.age = age
        self.gender = gender
    
    # Method to display the person's information
    def display_info(self):
        print(f"Name: {self.name}")
        print(f"Age: {self.age}")
        print(f"Gender: {self.gender}")
        
# Example usage
person1 = Person("John", 30, "Male")
person1.display_info()

person2 = Person("Jane", 25, "Female")
person2.display_info()

# Output: Name: John, Age: 30, Gender: Male
'''Q19) return JSON data in a Flask route'''
from flask import Flask, jsonify #importing flask and jsonify

app = Flask(__name__) #creating a Flask app

@app.route('/data') #creating a route
def data():
    return jsonify({"name": "John", "age": 30}) # A Python dictionary being returned as JSON data

if __name__ == '__main__':
    app.run() #running the app  

'''Q20) create a simple “Hello, World!” app in Flask'''
from flask import Flask

app = Flask(__name__) #creating a Flask app
@app.route('/') #creating a route (root url)
def hello_world():
    return "Hello World!" #returning a string

if __name__ == '__main__':
    app.run() #running the app

'''Q21) explain how Flask handles HTTP methods like GET and POST'''
# | Method   | Purpose                              |
# |----------|--------------------------------------|
# | `GET`    | Fetch data from the server (default) |
# | `POST`   | Send data to the server              |
# | `PUT`    | Update existing data                 |
# | `DELETE` | Remove data from the server          |

# Flask allows you to specify which HTTP methods a route should respond to by using the methods parameter in 
# the @app.route() decorator. By default, Flask routes respond to GET requests, but you can specify others 
# such as POST, PUT, DELETE, etc.

@app.route('/form', methods=['GET', 'POST'])
def handle_form():
    if request.method == 'POST': # If the route receives a POST request (usually from a form submission), it returns "Form submitted!" 
        return 'Form submitted!'
    return 'Form not yet submitted. 
# If it’s a GET request (user just visits the page), it returns "Form not yet submitted."

# Flask is a web framework written in Python.

# It helps Python developers build web applications more easily.

# You write your code in Python, and Flask provides tools to manage routes, forms, databases, and more.

'''Q22) Class to Represent a Person with Basic Attributes'''
class Person:
    # Constructor to initialize the attributes
    def __init__(self, name, age, gender): # The __init__ method is used to initialize the object's attributes (i.e., set up the initial state of the object).
        self.name = name # self refers to the current object
        self.age = age
        self.gender = gender
    
    # Method to display the person's information
    def display_info(self):
        print(f"Name: {self.name}")
        print(f"Age: {self.age}")
        print(f"Gender: {self.gender}")
        
# Example usage
person1 = Person("John", 30, "Male")
person1.display_info()

person2 = Person("Jane", 25, "Female")
person2.display_info()

# Output: Name: John, Age: 30, Gender: Male
'''Q19) return JSON data in a Flask route'''
from flask import Flask, jsonify #importing flask and jsonify

app = Flask(__name__) #creating a Flask app

@app.route('/data') #creating a route
def data():
    return jsonify({"name": "John", "age": 30}) # A Python dictionary being returned as JSON data

if __name__ == '__main__':
    app.run() #running the app  

'''Q20) create a simple “Hello, World!” app in Flask'''
from flask import Flask

app = Flask(__name__) #creating a Flask app
@app.route('/') #creating a route (root url)
def hello_world():
    return "Hello World!" #returning a string

if __name__ == '__main__':
    app.run() #running the app

'''Q21) explain how Flask handles HTTP methods like GET and POST'''
# | Method   | Purpose                              |
# |----------|--------------------------------------|
# | `GET`    | Fetch data from the server (default) |
# | `POST`   | Send data to the server              |
# | `PUT`    | Update existing data                 |
# | `DELETE` | Remove data from the server          |

# Flask allows you to specify which HTTP methods a route should respond to by using the methods parameter in 
# the @app.route() decorator. By default, Flask routes respond to GET requests, but you can specify others 
# such as POST, PUT, DELETE, etc.

@app.route('/form', methods=['GET', 'POST'])
def handle_form():
    if request.method == 'POST': # If the route receives a POST request (usually from a form submission), it returns "Form submitted!" 
        return 'Form submitted!'
    return 'Form not yet submitted. 
# If it’s a GET request (user just visits the page), it returns "Form not yet submitted."

# Flask is a web framework written in Python.

# It helps Python developers build web applications more easily.

# You write your code in Python, and Flask provides tools to manage routes, forms, databases, and more.

'''Q22) Class to Represent a Person with Basic Attributes'''
class Person:
    # Constructor to initialize the attributes
    def __init__(self, name, age, gender): # The __init__ method is used to initialize the object's attributes (i.e., set up the initial state of the object).
        self.name = name # self refers to the current object
        self.age = age
        self.gender = gender
    
    # Method to display the person's information
    def display_info(self):
        print(f"Name: {self.name}")
        print(f"Age: {self.age}")
        print(f"Gender: {self.gender}")
        
# Example usage
person1 = Person("John", 30, "Male")
person1.display_info()

person2 = Person("Jane", 25, "Female")
person2.display_info()

# Output: Name: John, Age: 30, Gender: Male
#         Name: Jane, Age: 25, Gender: Female

'''Q23) how hash table works, provide an example'''
# A hash table is a data structure that stores key-value pairs, allowing fast access, insertion, and deletion operations.

# How It Works:
# Hash Function: A function that converts the key into an index (hash) in an array.

# Collision Handling: When two keys hash to the same index, a collision occurs. 
# Common techniques are:
    # Chaining: Store multiple items at the same index using a list.
    # Open Addressing: Find another open index using a probing method.

# Simple hash table using a dictionary in Python
hash_table = {}

# Inserting key-value pairs
hash_table["name"] = "John"
hash_table["age"] = 30

# Retrieving values by key
print(hash_table["name"])  
# Output: John

'''Q24) First Non-Repeated Character in a String'''
def first_non_repeated_char(s):
    char_count = {}

    for char in s:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1

    for char in s:
        if char_count[char] == 1:
            return char

    return None

print(first_non_repeated_char("swiss")) 
print(first_non_repeated_char("aabbcc"))
# Output: w
#         None

'''Q25)  if Two Strings are Anagrams of Each Other'''
# An anagram is a word or phrase formed by rearranging the letters of 
# another word or phrase, using all the original letters exactly once.
def are_anagrams(str1, str2):
    if len(str1) != len(str2):
        return False
    
    return sorted(str1) == sorted(str2)

print(are_anagrams("listen", "silent"))  
print(are_anagrams("hello", "world"))
# Output: True
#         False