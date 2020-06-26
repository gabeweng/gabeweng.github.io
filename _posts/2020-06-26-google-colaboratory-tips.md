---
title: "Google Colab Tips for Power Users"
date: 2020-06-26T15:30-00:00
categories:
  - workflow
classes: wide
excerpt: Learn about lesser-known features in Google Colaboratory to improve your productivity.   
header:
  og_image: /images/colab-cover.png
  teaser: "/images/colab-cover.png"
---

Colab is one of the best products to come from Google. It has made GPUs freely accessible to learners and practitioners like me who otherwise wouldn't be able to afford a high-end GPU.  

While the interface is very easy to use, there are many lesser-known and undocumented features in colab. In this post, I will share features that I've discovered from their official talks and basic usage.    

## 1. Scratchpad Notebook
It's a pretty common scenario that we have a bunch of cluttered untitled notebooks created when we try out temporary stuff on colab.  
  
![](/images/colab-clutter.png){: .align-center}  
To solve this, you can bookmark the below link. It will open a special **scratch notebook** and any changes you make to that notebook are not saved to your main account.  

> [https://colab.research.google.com/notebooks/empty.ipynb](https://colab.research.google.com/notebooks/empty.ipynb)

## 2. Timing Execution of Cell  
It's pretty common that we manually calculate the difference between start and end times of a piece of code to gauge the time taken.  

Colab provides an inbuilt feature to do this. After a cell is executed, just hover over the cell run icon and you will get an estimate of the execution time taken.  

![](/images/colab-cell-hover.png){: .align-center}  

## 3. Run part of a cell  
You can also run only a part of the cell by selecting it and pressing the `Runtime > Run Selection` button or using the keyboard shortcut `Ctrl + Shift + Enter`.  

![](/images/colab-run-few-lines.gif){: .align-center}  

## 4. Jupyter Notebook Keyboard Shortcuts  
If you are familiar with keyboard shortcuts from Jupyter Notebook, they don't work directly in Colab. I found a mental model to map between them.  

Just add `Ctrl + M` before whatever keyboard shortcut you were using in Jupyter.  This rule of thumb works for the majority of common use-cases.  

|Action| Jupyter Notebook | Google Colab|
|---|---|---|
|Add a cell above| A | Ctrl + **M** + A|
|Add a cell below| B | Ctrl + **M** + B|
|See all keyboard shorcuts| H | Ctrl + **M** + H|
|Change cell to code| Y| Ctrl + **M** + Y|
|Change cell to markdown| M | Ctrl + **M** + M|
|Interrupt the kernel| II | Ctrl + **M** + I|
|Delete a cell| DD | Ctrl + **M** + D|
|Checkpoint notebook| Ctrl + S | Ctrl + **M** + S|

Below are some notable exceptions to this rule for which either the shortcut is changed completely or kept the same.   

|Action| Jupyter Notebook | Google Colab|
|---|---|---|
|Restart runtime| 00 | Ctrl + **M** + **.**|
|Run cell| Ctrl + Enter | Ctrl + Enter|
|Run cell and add new cell below| Alt + Enter | Alt + Enter|
|Run cell and goto the next cell below| Shift + Enter | Shift + Enter|
|Comment current line| Ctrl + / | Ctrl + /|

## 5. Jump to Class Definition  
Similar to an IDE, you can go to a class definition by pressing `Ctrl` and then clicking a class name. For example, here we view the class definition of the Dense layer in Keras by pressing Ctrl and then clicking the `Dense` class name.  

![](/images/colab-goto-class.gif){: .align-center}  

## 6. Open Notebooks from GitHub   
The Google Colab team provides an official chrome extension to open notebooks on GitHub directly on colab. You can install it from [here](https://chrome.google.com/webstore/detail/open-in-colab/iogfkhleblhcpcekbiedikdehleodpjo).

After installation, click the colab icon on any GitHub notebook to open it directly.  
![](/images/colab-from-github.png){: .align-center}  

Alternatively, you can also manually open any GitHub notebook by replacing `github.com` with `colab.research.google.com/github`.
> https://**github.com**/fastai/course-v3/blob/master/nbs/dl1/00_notebook_tutorial.ipynb

to
> https://**colab.research.google.com/github**/fastai/course-v3/blob/master/nbs/dl1/00_notebook_tutorial.ipynb

## 7. Run Flask apps from Colab  
With a library called [flask-ngrok](https://github.com/gstaff/flask-ngrok), you can easily expose a Flask web app running on colab to demo prototypes. First, you need to install `flask` and `flask-ngrok`.
```python
!pip install flask-ngrok flask==0.12.2
```
Then, you just need to pass your flask app object to `run_with_ngrok` function and it will expose a ngrok endpoint when the server is started.  
```python
from flask import Flask
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)

@app.route('/')
def hello():
    return 'Hello World!'

if __name__ == '__main__':
    app.run()
```

![](/images/colab-flask.png){: .align-center}  

You can try this out from the package author's [official example](https://colab.research.google.com/github/gstaff/flask-ngrok/blob/master/examples/flask_ngrok_example.ipynb) on Colab.  

## 8. Switch between Tensorflow versions  
You can easily switch between Tensorflow 1 and Tensorflow 2 using this magic flag.   
To switch to Tensorflow 1.15.2, use this command:
```python
%tensorflow_version 1.x
```
To switch to Tensorflow 2.2, run this command:  
```python
%tensorflow_version 2.x
```
You will need to restart the runtime for the effect to take place. Colab recommends using the pre-installed Tensorflow version instead of installing it from `pip` for performance reasons.  

## 9. Tensorboard Integration   
Colab also provides a magic command to use Tensorboard directly from the notebook. You just need to set the logs directory location using the ``--logdir`` flag. You can learn to use it from the [official notebook](https://colab.research.google.com/github/tensorflow/tensorboard/blob/master/docs/tensorboard_in_notebooks.ipynb).  
```python
%load_ext tensorboard
%tensorboard --logdir logs
```

![](/images/colab-tensorboard.png){: .align-center}  

## 10. Gauge resource limits    
Colab provides the following specs for their free and pro versions. Based on your use case, you can switch to the pro version at $10/month if you need a better runtime, GPU and memory.  

|Version|GPU|GPU Ram|RAM|Storage|CPU Cores|Idle Timeout|Maximum Runtime|
|---|---|---|---|---|---|---|---|
|Free|Tesla K80|11.44GB|13.7GB|37GB|2|90 min|12 hrs|
|Pro|Tesla P100|16GB|27.4GB|37GB|4|90 min|24 hrs|

You can view the GPU you have been assigned by running the following command
```shell
!nvidia-smi
```
For information on the CPU, you can run this command
```shell
!cat /proc/cpuinfo
```

Similarly, you can view the RAM capacity by running
```python
import psutil
ram_gb = psutil.virtual_memory().total / 1e9
print(ram_gb)
```

## 11. Use interactive shell  
There is no built-in interactive terminal in Colab. But you can use the `bash` command to try out shell commands interactively. Just run this command and you will get a interactive input.    
```shell
!bash
```

Now, you can run any shell command in the given input box.  

![](/images/colab-bash.png){: .align-center}  

To quit from the shell, just type `exit` in the input box.  
  
![](/images/colab-bash-exit.png){: .align-center}  


## References
- Timothy Novikoff, ["Making the most of Colab (TF Dev Summit '20)"](https://www.youtube.com/watch?v=pnClcwTCyc0)
- Gal Oshri, ["What's new in TensorBoard (TF Dev Summit '19)"](https://www.youtube.com/watch?v=xM8sO33x_OU)
