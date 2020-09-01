---
title: "VSCode on Google Colab"
date: 2020-09-01T19:44-00:00
permalink: /vscode-on-colab/
categories:
  - colab
excerpt: Learn how to setup and use VSCode as an IDE on Google Colab
header:
  og_image: /images/colab-vscode.png
  teaser: /images/colab-vscode.png
classes: wide
---

I recently discovered a way to setup VSCode on Google Colab and use it as an IDE to edit code and run experiments on the Colab VM. 

With this setup, you can still prototype in the Colab Notebook while also using VSCode for all the advantages of an IDE. Here is how you can replicate my setup.  

## Steps  
I have described the steps in details below. After going through all the steps, please use this [colab notebook](https://colab.research.google.com/drive/1yvUy5Gn9lPjmCQH6RjD_LvUO2NE0Z7RM?usp=sharing) to try it out directly. 

1. First, we will install the [code-server](https://github.com/cdr/code-server) package to run VSCode editor as a web app. Copy and run the following command on colab to install `code-server`.  

    ```
    !curl -fsSL https://code-server.dev/install.sh | sh
    ```

2. After the installation is complete, we will expose a random port `9000` to an external URL we can access using the `pyngrok` package. To install `pyngrok`, run  

    ```shell
    !pip install -qqq pyngrok
    ```

3. Then, run the following following command to get a public ngrok URL. This will be the URL we will use to access VSCode. 

    ```python
    from pyngrok import ngrok
    url = ngrok.connect(port=9000)
    print(url)
    ```

4. Now, we will start the VSCode server in the background at port 9000 without any authentication using the following command.

    ```
    !nohup code-server --port 9000 --auth none &
    ```

5. Now, you can access the VSCode interface at the URL you got from step 3. The interface and functionality is same as the desktop version of VSCode.  

![](/images/colab-vscode.png){:.align-center}  

## Tips
- You can use `Ctrl + Shift + P` to open a popup for various actions.
- The Colab interface is still usable to upload and download files and mount with Google Drive. So, you get the benefits of both a notebook and a code editor.  

## References
- [Code-Server FAQs](https://github.com/cdr/code-server/blob/v3.5.0/doc/FAQ.md)
- [pyngrok - a Python wrapper for ngrok](https://pyngrok.readthedocs.io/en/latest/)
