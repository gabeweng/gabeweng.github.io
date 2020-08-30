---
title: "Text Data Augmentation with MarianMT"
date: 2020-08-30T19:44-00:00
permalink: /back-translation/
categories:
  - nlp
  - data augmentation
excerpt: Learn how to use machine translation models in Hugging Face Transformers for data augmentation
header:
  og_image: /images/back-translation-marianmt.png
  teaser: /images/back-translation-marianmt.png
classes: wide
---


Hugging Face recently released [1008 translation models](https://huggingface.co/models?search=Helsinki-NLP%2Fopus-mt) for almost 140 languages on their model hub. 

These models were originally trained by [Jörg Tiedemann](https://researchportal.helsinki.fi/en/persons/j%C3%B6rg-tiedemann) of the [Language Technology Research Group at the University of Helsinki](https://blogs.helsinki.fi/language-technology/). The original models were trained on the [Open Parallel Corpus(OPUS)](http://opus.nlpl.eu/) using a neural machine translation framework called [MarianNMT](https://marian-nmt.github.io/).

In this post, I will explain how you can use the MarianMT models to perform data augmentation for text.  

## Back Translation    
We will use a data augmentation technique called "Back Translation". In this, we take an original text written in English. Then, we convert it into another language (eg. French) using MarianMT. We translate the French text back into English using MarianMT. We keep the back-translated English text if it is different from the original English sentence.

![Backtranslation with MarianMT](/images/back-translation-marianmt.png){: .align-center}


## Augmentation Process [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1J_KpNYj03gecT0p9s6YeDcDJHKgPn1Hh?usp=sharing)

First, we need to install Hugging Face transformers and Moses Tokenizers with the following command
```shell
pip install -U transformers 
pip install mosestokenizer
```

After installation, we can now import the MarianMT model and tokenizer.
```python
from transformers import MarianMTModel, MarianTokenizer
```

Then, we can create a initialize the model that can translate from English to Romance languages. This is a single model that can translate to any of the romance languages()
```python
target_model_name = 'Helsinki-NLP/opus-mt-en-ROMANCE'
target_tokenizer = MarianTokenizer.from_pretrained(target_model_name)
target_model = MarianMTModel.from_pretrained(target_model_name)
```

Similarly, we can initialize models that can translate Romance languages to English.
```python
en_model_name = 'Helsinki-NLP/opus-mt-ROMANCE-en'
en_tokenizer = MarianTokenizer.from_pretrained(en_model_name)
en_model = MarianMTModel.from_pretrained(en_model_name)
```

Next, I have written a helper function to translate a batch of text given the machine translation model, tokenizer and the target romance language.  
```python
def translate(texts, model, tokenizer, language="fr"):
    # Prepare the text data into appropriate format for the model
    template = lambda text: f"{text}" if language == "en" else f"{language}<< {text}"
    src_texts = [template(text) for text in texts]

    # Tokenize the texts
    encoded = tokenizer.prepare_translation_batch(src_texts)
    
    # Generate translation using model
    translated = model.generate(**encoded)

    # Convert the generated tokens indices back into text
    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
    
    return translated_texts
```

Next, we will prepare a function to use the above `translate()` function to perform back translation.
```python
def back_translate(texts, target_lang="fr", source_lang="en"):
    # Translate to target language
    fr_texts = translate(texts, target_model, target_tokenizer, 
                         language=target_lang)

    # Translate from target language back to source language
    back_translated_texts = translate(fr_texts, en_model, en_tokenizer, 
                                      language=source_lang)
    
    return back_translated_texts
```

Now, we can perform data augmentation using back-translation from English to Spanish on a list of sentences as shown below.
```python
en_texts = ['This is so cool', 'I hated the food', 'They were very helpful']

aug_texts = back_translate(en_texts, source_lang="en", target_lang="es")
print(aug_texts)
```

```python
["Yeah, it's so cool.", "It's the food I hated.", 'They were of great help.']
```

Similarly, we can perform augmentation using English to French as shown below with the exact same helper method.
```python
en_texts = ['This is so cool', 'I hated the food', 'They were very helpful']
aug_texts = back_translate(en_texts, source_lang="en", target_lang="fr")

print(aug_texts)
```

```python
["It's so cool.", 'I hated food.', "They've been very helpful."]
```

## Available Models  
Here are language codes for a subset of major romance language that you can use above.  

|Language|French|Spanish|Italian|Portuguese|Romanian|Catalan|Galician|Latin|
|---|---|---|---|---|---|---|---|
|**Code**|fr|es|it|pt|ro|ca|gl|la|

|Language|Walloon|Occitan (post 1500)|Sardinian|Aragonese|Corsican|Romansh|
|---|---|---|---|---|---|---|
|**Code**|wa|oc|sn|an|co|rm|

To view all available language codes, you can run
```python
target_tokenizer.supported_language_codes
```

## Conclusion
Thus, MarianMT is a decent free and offline alternative to Google Translate for back-translation.  

## References
- [MarianMT - transformers 3.0.2 documentation](https://huggingface.co/transformers/model_doc/marian.html)
