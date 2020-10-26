---
title: "Machine Learning Cookbook"
permalink: /cookbook/
date: 2020-10-26T00:00-00:00
excerpt: Code snippets for repetitive tasks in Machine Learning
toc: true
toc_sticky: true
---

This is a personal collection of repetitive commands and snippets for ML projects.

## Conda
**Install OpenCV in conda**  
```shell
conda install -c conda-forge open-cv
```

**Update conda**  
```shell
conda update -n base -c defaults conda
```

**Create/Update conda environment from file**  
```shell
conda env create -f environment.yml
conda env update -f environment.yml
```

**Install CUDA toolkit in conda**  
```shell
conda install cudatoolkit=9.2 -c pytorch
conda install cudatoolkit=10.0 -c pytorch
```

## Celery
**Run celery workers**  
File `tasks.py` contains celery object, concurrency is set to 1 and no threads or process are used with `-P solo`

```shell
celery -A tasks.celery worker --loglevel=info --concurrency=1 -P solo
```


## Docker
**Dockerfile for FastAPI**  
{% gist ac87e33d8522f5734e5b9c085bca86ae %}

## Git
**Prevent git from asking for password**  
```shell
git config credential.helper 'cache --timeout=1800'
```

**Whitelist in .gitignore**  
```shell
# First, ignore everything
*

# Whitelist all directory
!*/

# Only .py and markdown files
!*.py
!*.md
!*.gitignore
```

**Clone private repo using personal token**  

Create token from [settings](https://github.com/settings/tokens) and run:
```shell
git clone https://<token>@github.com/amitness/example.git
```

**Create alias to run command**  
```shell
# git test
git config --global alias.test "!python -m doctest``"
```

## Gunicorn
**Increase timeout**  
```shell
gunicorn --bind 0.0.0.0:5000 app:app --timeout 6000
```

**Check error logs**  
```shell
tail -f /var/log/gunicorn/error_
```

**Run two workers**  
```shell
gunicorn app:app  --preload -w 2 -b 0.0.0.0:5000
```

## Jupyter Notebook
**Auto-import common libraries**  
1. Create `startup` folder in `~/.ipython/profile_default`
2. Create a python file `start.py`
3. Add imports there.

```python
# start.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

**Add conda kernel to jupyter**  
Activate conda environment and run below command.

```shell
pip install --user ipykernel
python -m ipykernel install --user --name=condaenv
```

**Start notebook on remote server**  
```shell
jupyter notebook --ip=0.0.0.0 --no-browser
```

**Serve as voila app**  
```shell
voila --port=$PORT --no-browser app.ipynb
```

## Linux
**Use remote server as VPN**  
```shell
ssh -D 8888 -f -C -q -N ubuntu@example.com
```

**SSH Tunneling for multiple ports (5555, 5556)**  
```shell
ssh -N -f -L localhost:5555:127.0.0.1:5555 -L localhost:5556:127.0.0.1:5556 ubuntu@example.com
```

**Reverse SSH tunneling**  
Enable `GatewayPorts=yes` in `/etc/ssh/sshd_config` on server.

```shell
ssh -NT -R example.com:5000:localhost:5000 ubuntu@example.com -i ~/.ssh/xyz.pem -o GatewayPorts=yes
```

**Copy remote files to local**  
```shell
scp ubuntu@example.com:/mnt/file.zip .
```

## Nginx  
**Assign path to port**  
```shell
location /demo/ {
                proxy_pass http://localhost:5000/;
                proxy_http_version 1.1;
                proxy_set_header Upgrade $http_upgrade;
                proxy_set_header Connection "upgrade";
	}
```

**Increase timeout for nginx**  
Default timeout is 60s. Run below command or use [alternative](https://blogs.agilefaqs.com/tag/proxy_read_timeout/).

```
sudo nano /etc/nginx/proxy_params
```

```shell
proxy_set_header Host $http_host;
proxy_set_header X-Real-IP $remote_addr;
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
proxy_set_header X-Forwarded-Proto $scheme;
proxy_connect_timeout   300;
proxy_send_timeout      300;
proxy_read_timeout      300;
```

## Pandas
**Save with quoted strings**  
```
df.to_csv('data.csv', 
            index=False, 
            quotechar='"',
            quoting=csv.QUOTE_NONNUMERIC))
```

## Python
**Send email with SMTP**  
{% gist 7a806edddd79f2e7e0e743ec756ca4e5 %}

**Run selenium on chromium**  
```shell
sudo apt-get update
sudo apt install chromium-chromedriver
cp /usr/lib/chromium-browser/chromedriver /usr/bin
pip install selenium
```

```python
from selenium import webdriver

# set options to be headless:
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome('chromedriver',options=options)
```

## PyTorch
**Install CPU-only version of PyTorch**  
```
conda install pytorch torchvision cpuonly -c pytorch
```

**Set random seed**  
```python
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## Pytorch Lightning
**Use model checkpoint callback**  
{% gist a64feca21ea6afbe3e873dbdbed7f384 %}

## Redis
**Connect to redis from commandline**
```shell
redis-cli -h 1.1.1.1 -p 6380 -a password
```

## Requests
**Post JSON data to endpoint**  
```python
import json
import requests

headers = {'Content-Type': 'application/json'}
data = {}
response = requests.post('http://example.com',
                         data=json.dumps(data),
                         headers=headers)
```

## SSH
**Add server alias to SSH config**  
Add to `~/.ssh/config`

{% gist edf6f3333146eb6ab2df0113f23dd837 %}

## Streamlit
**Disable CORS**  
Create `~/.streamlit/config.toml`

```shell
[server]
enableCORS = false
```

**File Uploader**  
```python
file = st.file_uploader("Upload file", 
                        type=['csv', 'xlsx'], 
                        encoding='latin-1')
df = pd.read_csv(file)
```

**Create download link for CSV file**  
```python
import base64

csv = df.to_csv(index=False)
filename = 'data.csv'
b64 = base64.b64encode(csv.encode()).decode()
href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download CSV File</a>'
st.markdown(href, unsafe_allow_html=True)
```

**Run on docker**  
```
FROM python:3.7
EXPOSE 8501
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
CMD streamlit run src/main.py
```

**Run on heroku**  
Add `requirements.txt`, create [Procfile](https://github.com/arvkevi/nba-roster-turnover/blob/master/Procfile) and [setup.sh](https://github.com/arvkevi/nba-roster-turnover/blob/master/setup.sh).


**Render SVG**  
```python
import base64
import streamlit as st

def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = f'<img src="data:image/svg+xml;base64,{b64}"/>'
    st.write(html, unsafe_allow_html=True)
```


## Tensorflow
**Install CPU-only version of Tensorflow**

```shell
conda install tensorflow-mkl
```
or
```shell
pip install tensorflow-cpu==2.1.0
```

**Install custom builds for CPU**  

Find link from [https://github.com/lakshayg/tensorflow-build](https://github.com/lakshayg/tensorflow-build)

```shell
pip install --ignore-installed --upgrade "url"
```

**Use only single GPU**  
```shell
export CUDA_VISIBLE_DEVICES=0
```

**Allocate memory as needed**  
```shell
export TF_FORCE_GPU_ALLOW_GROWTH='true'
```

**Load saved model with custom layer**  
```python
from tensorflow.keras.models import load_model
import tensorflow_hub as hub

model = load_model(model_name, 
                   custom_objects={'KerasLayer':hub.KerasLayer})
```

**Ensure Conda doesn't cause tensorflow issue**  
{% gist 5c63f2101652ecbeadf5e5c505d40922 %}















