---
title: "FastAPI for Flask Users"
date: 2020-06-29T12:00-00:00
categories:
  - serving
excerpt: A comprehensive guide to FastAPI with a side-by-side code comparison with Flask    
header:
  og_image: /images/flask-to-fastapi.png
  teaser: "/images/flask-to-fastapi.png"
toc: true
toc_sticky: true
---

While Flask has become the de-facto choice for API development in Machine Learning projects, there is a new framework called FastAPI that has been getting a lot of community traction.  

![](/images/flask-to-fastapi.png){: .align-center}  

I recently decided to give FastAPI a spin by porting a production project written in Flask. FastAPI was very easy to pick up coming from Flask and I was able to get things up and running in just a few hours. 

The added benefit of automatic data validation, documentation generation and baked-in best-practices such as pydantic schemas and python typing makes this a strong choice for future projects.  

In this post, I will introduce FastAPI by contrasting the implementation of various common use-cases in both Flask and FastAPI.  

<div class="notice--info">
<strong>Version Info:</strong>
<p>
At the time of this writing, the Flask version is 1.1.2 and the FastAPI version is 0.58.1
</p>
</div>

## Installation  
Both Flask and FastAPI are available on PyPI. For conda, you need to use the `conda-forge` channel to install FastAPI while it's available in the default channel for Flask.  

**Flask:**
```shell
pip install flask
conda install flask
```

**FastAPI:**
```shell
pip install fastapi uvicorn
conda install fastapi uvicorn -c conda-forge
```

## Running "Hello World" 
**Flask:**
```python
# app.py
from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return {'hello': 'world'}

if __name__ == '__main__':
    app.run()
```

Now you can run the development server using the below command. It runs on port 5000 by default.  
```shell
python app.py
```

**FastAPI**  
```python
# app.py
import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def home():
    return {'hello': 'world'}

if __name__ == '__main__':
    uvicorn.run(app)
```

FastAPI defers serving to a production-ready server called `uvicorn`. We can run it in development mode with a default port of 8000.   
```shell
python app.py
```

## Production server 
**Flask:**
```python
# app.py
from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return {'hello': 'world'}

if __name__ == '__main__':
    app.run()
```

For a production server, `gunicorn` is a common choice in Flask.  
```shell
gunicorn app:app
```

**FastAPI**  
```python
# app.py
import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def home():
    return {'hello': 'world'}

if __name__ == '__main__':
    uvicorn.run(app)
```

FastAPI defers serving to a production-ready server called `uvicorn`. We can start the server as:
```shell
uvicorn app:app
```

You can also start it in hot-reload mode by running
```shell
uvicorn app:app --reload
```

## HTTP Methods  
**Flask:**
```python
@app.route('/', methods=['POST'])
def example():
    ...
```

**FastAPI:**  
```python
@app.post('/')
def example():
    ...
```
You have individual decorator methods for each HTTP method.
```python
@app.get('/')
@app.put('/')
@app.patch('/')
@app.delete('/')
```

## URL Variables  
We want to get the user id from the URL e.g. `/users/1` and then return the user id to the user.  
 
**Flask:**  
```python
@app.route('/users/<int:user_id>')
def get_user_details(user_id):
    return {'user_id': user_id}
```

**FastAPI:**  

In FastAPI, we make use of type hints in Python to specify all the data types. For example, here we specify that `user_id` should be an integer. The variable in the URL path is also specified similar to f-strings.  

```python
@app.get('/users/{user_id}')
def get_user_details(user_id: int):
    return {'user_id': user_id}
```

## Query Strings    
We want to allow the user to specify a search term by using a query string `?q=abc` in the URL.  
 
**Flask:**  
```python
from flask import request

@app.route('/search')
def search():
    query = request.args.get('q')
    return {'query': query}
```

**FastAPI:**  
```python
@app.get('/search')
def search(q: str):
    return {'query': q}
```

## JSON POST Request  
Let's take a toy example where we want to send a JSON POST request with a `text` key and get back a lowercased version.  
```json
# Request
{"text": "HELLO"}

# Response
{"text": "hello"}
```

 
**Flask:**  
```python
from flask import request

@app.route('/lowercase', methods=['POST'])
def lower_case():
    text = request.json.get('text')
    return {'text': text.lower()}
```

**FastAPI:**  
If you simply replicate the functionality from Flask, you can do it as follows in FastAPI.  
```python
from typing import Dict

@app.post('/lowercase')
def lower_case(json_data: Dict):
    text = json_data.get('text')
    return {'text': text.lower()}
```

But, this is where FastAPI introduces a new concept of creating Pydantic schema that maps to the JSON data being received. We can refactor the above example using pydantic as:   
```python
from pydantic import BaseModel

class Sentence(BaseModel):
    text: str

@app.post('/lowercase')
def lower_case(sentence: Sentence):
    return {'text': sentence.text.lower()}
```

As seen, instead of getting a dictionary, the JSON data is converted into an object of the schema `Sentence`. As such, we can access the data using data attributes such as `sentence.text`. This also provides automatic validation of data types. If the user tries to send any data other than a string, they will be given an auto-generated validation error.  
  
**Example Invalid Request**
```json
{"text": null}
```

**Automatic Response**
```json
{
    "detail": [
        {
            "loc": [
                "body",
                "text"
            ],
            "msg": "none is not an allowed value",
            "type": "type_error.none.not_allowed"
        }
    ]
}
```

## Separated Views  
We want to decompose the views from a single app.py into separate files.
```json
- app.py
- views
  - user.py
``` 
 
**Flask:**  
In Flask, we use a concept called blueprints to manage this. We would first create a blueprint for the user view as:
```python
# views/user.py
from flask import Blueprint
user_blueprint = Blueprint('user', __name__)

@user_blueprint.route('/users')
def list_users():
    return {'users': ['a', 'b', 'c']}

```
Then, this view is registered in the main `app.py` file.
```python
# app.py
from flask import Flask
from views.user import user_blueprint

app = Flask(__name__)
app.register_blueprint(user_blueprint)
```

**FastAPI:**  
In FastAPI, the equivalent of blueprint is called router. First, we create a user router as:
```python
# routers/user.py
from fastapi import APIRouter
router = APIRouter()

@router.get('/users')
def list_users():
    return {'users': ['a', 'b', 'c']}
```

Then, we attach this router to the main app object as:  
```python
# app.py
from fastapi import FastAPI
from routers import user

app = FastAPI()
app.include_router(user.router)
```

## Data Validation  
**Flask**  
Flask doesn't provide any input data validation feature out-of-the-box. It's common practice to either write your own validation logic or use libraries such as [marshmalllow](https://marshmallow.readthedocs.io/en/stable/) or [pydantic](https://pydantic-docs.helpmanual.io/).

**FastAPI:**  

FastAPI wraps pydantic into its framework and allows data validation by simply using a combination of pydantic schema and python type hints.  

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    name: str
    age: int

@app.post('/users')
def save_user(user: User):
    return {'name': user.name,
            'age': user.age}
```

This code will perform automatic validation to ensure `name` is string and `age` is integer. If any other data type is sent, it auto generates validation error with relevant message.  

Here are some examples of pydantic schema for common usecases.  
 
### Example 1: Key-value pairs
```json
{
  "name": "Isaac",
  "age": 60
}
```
```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
```

### Example 2: Collection of things  
```json
{
  "series": ["GOT", "Dark", "Mr. Robot"]
}
```
```python
from pydantic import BaseModel
from typing import List

class Metadata(BaseModel):
    series: List[str]
```

### Example 3: Nested Objects  
```json
{
  "users": [
    {
      "name": "xyz",
      "age": 25
    },
    {
      "name": "abc",
      "age": 30
    }
  ],
  "group": "Group A"
}
```
```python
from pydantic import BaseModel
from typing import List

class User(BaseModel):
    name: str
    age: int

class UserGroup(BaseModel):
    users: List[User]
    group: str
```

## Conclusion  
Thus, FastAPI is a good alternative to Flask for building robust APIs with all the best-practices baked in.  

## References
- [FastAPI Documentation](https://fastapi.tiangolo.com)