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

**Disable auto-activation of conda environment**
```shell
conda config --set auto_activate_base false
```

## Celery
**Run celery workers**  
File `tasks.py` contains celery object, concurrency is set to 1 and no threads or process are used with `-P solo`

```shell
celery -A tasks.celery worker --loglevel=info --concurrency=1 -P solo
```

## Colab
**Force remount google drive**  
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```


## Docker
**Start docker-compose as daemon**
```shell
docker-compose up --build -d
```

**Disable pip cache and version check**
```shell
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
```

**Dockerfile for FastAPI**  
{% gist ac87e33d8522f5734e5b9c085bca86ae %}

## FastAPI
**Use debugging mode**  
```python
# server.py

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
```

**Enable CORS**  
{% gist ec3e27028aa5d46f45fd0aeb435723f1 %}

**Run FastAPI in Jupyter Notebook**  
{% gist 5702a2c3787749a783360c0129133408 %}

## Flask
**Test API in flask**
{% gist ff5579911c0eb7d4aa193596e02b9135 %}

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

**Install Git LFS**  
```shell
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
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

## Huey  
**Add background task to add 2 numbers**  
{%gist c2ec5221c1d759c43f77111c438d13e5 %}

## Jupyter
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

**Auto print all expressions**  
Edit `~/.ipython/profile_default/ipython_config.py` and add
```
c = get_config()

# Run all nodes interactively
# c.InteractiveShell.ast_node_interactivity = "last_expr"
c.InteractiveShell.ast_node_interactivity = "all"
```

**Add conda kernel to jupyter**  
Activate conda environment and run below command.

```shell
pip install --user ipykernel
python -m ipykernel install --user --name=condaenv
```

**Add R kernel to jupyter**    

```shell
conda install -c r r-irkernel

# Link to fix issue with readline
cd /lib/x86_64-linux-gnu/
sudo ln -s libreadline.so.7.0 libreadline.so.6
```

**Start notebook on remote server**  
```shell
jupyter notebook --ip=0.0.0.0 --no-browser
```

**Serve as voila app**  
```shell
voila --port=$PORT --no-browser app.ipynb
```

**Enable widgets in jupyter lab**  
```shell
pip install jupyterlab
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

**Switch to language server in jupyter lab**  
```shell
pip install — pre jupyter-lsp
jupyter labextension install @krassowski/jupyterlab-lsp
pip install python-language-server[all]
```

## Kaggle
**Add kaggle credentials**  
```shell
pip install --upgrade  kaggle kaggle-cli

mkdir ~/.kaggle
mv kaggle.json ~/.kaggle
chmod 600 /root/.kaggle/kaggle.json
```

## Linux
**Zip a folder**
```shell
zip -r folder.zip folder
```

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

**Set correct permission for PEM file**
```shell
chmod 400 credentials.pem
```

**Clear DNS cache**
```shell
sudo service network-manager restart
sudo service dns-clean
sudo systemctl restart dnsmasq
sudo iptables -F
```

**Unzip .xz file**  
```shell
sudo apt-get install xz-utils
unxz ne.txt.xz
```

**Disable password-based login on server**  
- Edit this file and set `PasswordAuthentication` to `no`
```shell
sudo nano /etc/ssh/sshd_config
```

**Auto-generate help for make files**  
{% gist 14ba10992f7437cfac2b7a64a8f0a67a %}

## Markdown
**Add comparison of code blocks side by side**  
[Solution](https://stackoverflow.com/a/59314488/10137343)

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

**Setup nginx for prodigy**  
{% gist 2cc8f79a12a89e6cd22857c5bab49c8c %}

## NLTK  
**Get list of all POS tags**  
```python
import nltk
nltk.download('tagsets')
nltk.help.upenn_tagset()
```

## NPM
**Upgrade to latest node version**
```shell
npm cache clean -f
npm install -g n 
n stable
```

## Pandas
**Save with quoted strings**  
```python
df.to_csv('data.csv', 
            index=False, 
            quotechar='"',
            quoting=csv.QUOTE_NONNUMERIC)
```

## Postgres  
**Import database dump**  
If database name is `test` and user is `postgres`.  

```shell
pg_restore -U postgres -d test < example.dump
```

## Pycharm
**Add keyboard shortcut for custom command**  
[Link](https://intellij-support.jetbrains.com/hc/en-us/community/posts/207070295/comments/207021015)

**Enable pytest as default test runner**  
![](/images/pytest-pycharm.png){:.align-center}  

## Pydantic
**Allow camel case field name from frontend**  
{%gist 97f5d44088ddf5ffa97f0cdb4b3360ba %}

**Validate fields**  
{% gist f9cc1f2afa48762aad5352aeb3043777 %}

## Python
**Install build utilities**  
```shell
sudo apt update
sudo apt install build-essential python3-dev
sudo apt install python-pip virtualenv
```

**Install mysqlclient**  
```shell
sudo apt-get install libmysqlclient-dev mysql-server
pip install mysqlclient
```

**Convert python package to command line tool**  
{% gist a2889d272afdd5a454230f00d2079104 %}

{% gist 6883909479adb11c0f3fee924175f21d %}

**Send email with SMTP**  
- Enable `less secure app access` in [settings](https://myaccount.google.com/lesssecureapps) of gmail.
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

**Generate fake user agent in selenium**  
Run `pip install fake_useragent`.

```python
from fake_useragent import UserAgent
from selenium import webdriver

ua = UserAgent(verify_ssl=False)
user_agent = ua.random

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument(f"user-agent={user_agent}")
driver = webdriver.Chrome(chrome_options=chrome_options)
```

## PyTorch
**Install CPU-only version of PyTorch**  
```
conda install pytorch torchvision cpuonly -c pytorch
```

**Auto-select proper pytorch version based on GPU**  
```
pip install light-the-torch
ltt install torch torchvision
```

**Set random seed**  
```python
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

**Create custom transformation**  
```python
class Reshape:
    def __init__(self, new_shape):
        self.new_shape = new_shape

    def __call__(self, img):
        return torch.reshape(img, self.new_shape)
```

## Pytorch Lightning
**Use model checkpoint callback**  
{% gist a64feca21ea6afbe3e873dbdbed7f384 %}

## Redis
**Connect to redis from commandline**
```shell
redis-cli -h 1.1.1.1 -p 6380 -a password
```

**Connect to local redis**  
```python
from redis import Redis
conn = Redis(host='127.0.0.1', port='6379')
conn.set('age', 100)
```

**Add password to redis server**  
  
Edit `/etc/redis/redis.conf` file.  
```shell
sudo nano /etc/redis/redis.conf
```

Uncomment this line and set password.
```shell
# requirepass yourpassword
```

Restart redis server.
```shell
sudo service redis-server restart
```

**Enable cluster mode locally**  
- Edit `/etc/redis/redis.conf` file.  
```shell
sudo nano /etc/redis/redis.conf
```

- Uncomment this line and save file.
{% gist cf2ac62e38d2e20ff06e52ac49ae58df %}

- Restart redis server.
```shell
sudo service redis-server restart
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

**Use random user agent in requests**  
{% gist f97ad3502d2e7321e0a38ac10bb11df2 %}

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

**Docker compose for streamlit**

![](/images/streamlit-docker-compose.png){:.align-center}

Add `Dockerfile` to app folder.
{% gist 6573c17effd3c6245ace31994245a1c1 %}

Add `project.conf` to nginx folder.
{% gist 632911fe32f11f859799344baa196e9e %}

Add `Dockerfile` to nginx folder.
{% gist bf8cc293f79ef414bb0a2b611b483034 %}

Add `docker-compose.yml` at the root
{% gist 375deb78641bc23c79b5127d668fcb03 %}

**Run on heroku**  
Add `requirements.txt`, create [Procfile](https://github.com/arvkevi/nba-roster-turnover/blob/master/Procfile) and [setup.sh](https://github.com/arvkevi/nba-roster-turnover/blob/master/setup.sh).

**Deploy streamlit on google cloud**  
Create [Dockerfile](https://github.com/Jcharis/Streamlit_DataScience_Apps/blob/master/Deploying_Streamlit_Apps_To_GCP/Dockerfile), [app.yaml](https://github.com/Jcharis/Streamlit_DataScience_Apps/blob/master/Deploying_Streamlit_Apps_To_GCP/app.yaml) and run:
```shell
gcloud config set project your_projectname
gcloud app deploy
```

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

**Install with GPU support**  
```shell
conda create --name tensorflow-22 \
    tensorflow-gpu=2.2 \
    cudatoolkit=10.1 \
    cudnn=7.6 \
    python=3.8 \
    pip=20.0
```

**Use only single GPU**  
```shell
export CUDA_VISIBLE_DEVICES=0
```

**Allocate memory as needed**  
```shell
export TF_FORCE_GPU_ALLOW_GROWTH='true'
```

**Enable XLA**  
```python
import tensorflow as tf
tf.config.optimizer.set_jit(True)
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

**Upload tensorboard data to cloud**
```shell
tensorboard dev upload --logdir ./logs \
    --name "XYZ" \
    --description "some model"
```

**Use TPU in Keras**  
[TPU survival guide on Google Colaboratory](https://maelfabien.github.io/bigdata/ColabTPU/#connect-the-tpu-and-test-it)

**Use universal sentence encoder**  
```python
import tensorflow_hub as hub
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
e = embed(['hello', 'hey']).numpy()
```

## Textblob
**Backtranslate a text**  
```python
from textblob import TextBlob

def back_translate(text):
    t = TextBlob(text)
    return (TextBlob(t.translate('en', 'zh').raw)
            .translate('zh', 'en')
            .raw)
```
