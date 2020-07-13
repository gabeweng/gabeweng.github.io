---
title: "Machine Learning Toolbox"
permalink: /toolbox/
date: 2020-07-10T12:00-00:00
excerpt: A curated list of libraries for all phases of the Machine Learning workflow   
header:
  og_image: /images/toolbox.png
  teaser: "/images/toolbox.png"
toc: true
toc_sticky: true
---

This page contains useful libraries I've found when working on Machine Learning projects.  

The libraries are organized below by phases of a typical Machine Learning project.  

## Phase: Data
### Data Annotation  

|Category|Tool|Remarks|
|---|---|---|
|Image | [makesense.ai](https://www.makesense.ai/)||
|Text | [doccano](https://doccano.herokuapp.com/), [dataturks](https://dataturks.com/), [brat](http://brat.nlplab.org/)||
| | [prodigy](https://prodi.gy/)|Paid|
|Audio | [audio-annotator](https://github.com/CrowdCurio/audio-annotator), [audiono](https://github.com/midas-research/audino)||
|General| [superintendent](https://superintendent.readthedocs.io/en/latest/installation.html)|Label in notebooks|
|| [labelstudio](https://labelstud.io/)|Open Source Data Labeling Tool|

### Data Collection  

|Category|Tool|Remarks|
|---|---|---|
|Curations|[nlp-datasets](https://github.com/niderhoff/nlp-datasets), [The Big Bad NLP Database](https://quantumstat.com/dataset/dataset.html), [datasetlist](https://www.datasetlist.com/), [UCI](https://archive.ics.uci.edu/ml/datasets.php), [Google Dataset Search](https://toolbox.google.com/datasetsearch), [fastai-datasets](https://course.fast.ai/datasets.html)||
|Words|[curse-words](https://github.com/reimertz/curse-words), [badwords](https://github.com/MauriceButler/badwords), [LDNOOBW](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words), [10K most common words](https://github.com/first20hours/google-10000-english), [common-misspellings](https://bitbucket.org/bedizel/moe/src/master/data/)||
||[english-words](https://github.com/dwyl/english-words)|A text file containing over 466k English words|
|Text Corpus|[project gutenberg](https://www.gutenberg.org/), [oscar (big multilingual corpus)](https://traces1.inria.fr/oscar/), [nlp-datasets](https://github.com/niderhoff/nlp-datasets),  [1 trillion n-grams](https://catalog.ldc.upenn.edu/LDC2006T13), [litbank](https://github.com/dbamman/litbank), [BookCorpus](https://github.com/soskek/bookcorpus), [south-asian text corpus](https://github.com/google-research-datasets/dakshina)||
|Sentiment|[SST2](https://github.com/clairett/pytorch-sentiment-classification/tree/master/data/SST2), [Amazon Reviews](https://www.kaggle.com/bittlingmayer/amazonreviews), [Yelp Reviews](https://www.kaggle.com/yelp-dataset/yelp-dataset), [Movie Reviews](http://www.cs.cornell.edu/people/pabo/movie-review-data/), [Food Reviews](https://www.kaggle.com/snap/amazon-fine-food-reviews), [Twitter Airline](https://www.kaggle.com/crowdflower/twitter-airline-sentiment), [GOP Debate](https://www.kaggle.com/crowdflower/first-gop-debate-twitter-sentiment), [Sentiment Lexicons for 81 languages](https://www.kaggle.com/rtatman/sentiment-lexicons-for-81-languages), [SentiWordNet](http://sentiwordnet.isti.cnr.it/), [Opinion Lexicon](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon), [Wordstat words](https://provalisresearch.com/products/content-analysis-software/wordstat-dictionary/sentiment-dictionaries/), [Emoticon Sentiment](http://people.few.eur.nl/hogenboom/files/EmoticonSentimentLexicon.zip)||
|Emotion|[NRC-Emotion-Lexicon-Wordlevel](https://raw.githubusercontent.com/dinbav/LeXmo/master/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt), [ISEAR(17K)](https://github.com/PoorvaRane/Emotion-Detector/blob/master/ISEAR.csv), [HappyDB](https://megagon.ai/projects/happydb-a-happiness-database-of-100000-happy-moments/)|
|Summarization|[curation-corpus](https://github.com/CurationCorp/curation-corpus)||
|Conversations|[conversational-datasets](https://github.com/PolyAI-LDN/conversational-datasets), [cornell-movie-dialog-corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)||
|Image|[1 million fake faces](https://archive.org/details/1mFakeFaces), [flickr-faces](https://github.com/NVlabs/ffhq-dataset), [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), [SVHN](http://ufldl.stanford.edu/housenumbers/), [STL-10](http://ai.stanford.edu/~acoates/stl10/), [imagenette](https://github.com/fastai/imagenette), [objectnet](https://objectnet.dev/), [YFCC100m](http://projects.dfki.uni-kl.de/yfcc100m/), [USPS](https://www.kaggle.com/bistaumanga/usps-dataset), [Animal Faces-HQ dataset (AFHQ)](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq)||
|Paraphrasing| [PPDB](http://paraphrase.org/)||
|One Shot Learning | [omniglot](https://github.com/brendenlake/omniglot), [mini-imagenet](https://github.com/yaoyao-liu/mini-imagenet-tools)|
|Audio | [audioset](https://research.google.com/audioset/index.html)|YouTube audio with labels|
|Graphs| [Social Networks (Github, Facebook, Reddit)](https://github.com/benedekrozemberczki/datasets)||
|Handwriting| [iam-handwriting](http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database)||

### Importing Data  

|Category|Tool|Remarks|
|---|---|---|
| Prebuilt| [openml](https://openml.github.io/openml-python/master/), [lineflow](https://github.com/tofunlp/lineflow)||
||[rs_datasets](https://darel13712.github.io/rs_datasets/)|Recommendation Datasets|
||[nlp](https://github.com/huggingface/nlp)|Python interface to NLP datasets|
| Audio| [pydub](https://github.com/jiaaro/pydub)||
| Video| [moviepy](https://zulko.github.io/moviepy/)|Edit Videos|
| | [pytube](https://github.com/nficano/pytube)|Download youtube vidoes|
| Image| [py-image-dataset-generator](https://github.com/tomahim/py-image-dataset-generator)|Auto fetch images from web for certain search|
| News| [news-please](https://github.com/fhamborg/news-please), [news-catcher](https://github.com/kotartemiy/newscatcher/blob/master/README.md)|Scrap News|
|| [pygooglenews](https://github.com/kotartemiy/pygooglenews)|Google News|
| Lyrics| [lyricsgenius](https://github.com/johnwmillr/LyricsGenius)||
| Email| [talon](https://github.com/mailgun/talon)||
| PDF| [camelot](https://camelot-py.readthedocs.io/en/master/), [tabula-py](https://github.com/chezou/tabula-py), [parsr](https://github.com/axa-group/Parsr), [pdftotext](https://pypi.org/project/pdftotext/), [pdfplumber](https://github.com/jsvine/pdfplumber)||
| Excel| [openpyxl](https://openpyxl.readthedocs.io/en/stable/)||
| Remote file| [smart_open](https://github.com/RaRe-Technologies/smart_open)||
| Crawling| [MechanicalSoup](https://github.com/MechanicalSoup/MechanicalSoup), [libextract](https://github.com/datalib/libextract)||
| | [pyppeteer](https://github.com/pyppeteer/pyppeteer)|Chrome Automation|
| Google sheets| [gspread](https://github.com/burnash/gspread)||
| Google drive| [gdown](https://github.com/wkentaro/gdown), [pydrive](https://pythonhosted.org/PyDrive/index.html)||
| Python API| [pydataset](https://github.com/iamaziz/PyDataset)||
| Google Maps| [geo-heatmap](https://github.com/luka1199/geo-heatmap)||
| Text to Speech| [gtts](https://github.com/pndurette/gTTS)||
| Database| [blaze](https://github.com/blaze/blaze)|Pandas and Numpy interface to databases|
| Twitter| [twint](https://github.com/twintproject/twint)|Scrape Twitter|
| App Store| [google-play-scraper](https://github.com/JoMingyu/google-play-scraper)||

### Data Augmentation  

|Category|Tool|Remarks|
|---|---|---|
| Text| [nlpaug](https://github.com/makcedward/nlpaug), [noisemix](https://github.com/noisemix/noisemix), [textattack](https://github.com/QData/TextAttack), [textaugment](https://github.com/dsfsi/textaugment), [niacin](https://github.com/deniederhut/niacin)||
| Image| [imgaug](https://github.com/aleju/imgaug/), [albumentations](https://github.com/albumentations-team/albumentations), [augmentor](https://github.com/mdbloice/Augmentor), [solt](https://mipt-oulu.github.io/solt/index.html)||
| Audio| [audiomentations](https://github.com/iver56/audiomentations), [muda](https://github.com/bmcfee/muda)||
| OCR data| [TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator)||
| Tabular data| [deltapy](https://github.com/firmai/deltapy)||
| Automatic augmentation| [deepaugment](https://pypi.org/project/deepaugment/)|Image|

## Phase: Exploration

###  Data Preparation  

|Category|Tool|Remarks|
|---|---|---|
| Dataframe| [cudf](https://github.com/rapidsai/cudf)|Pandas on GPU|
| Missing values| [missingno](https://github.com/ResidentMario/missingno)||
| Split images into train/validation/test| [split-folders](https://github.com/jfilter/split-folders)||
| Class Imbalance| [imblearn](https://imbalanced-learn.readthedocs.io/en/stable/api.html)||
| Categorical encoding| [category_encoders](https://github.com/scikit-learn-contrib/category_encoders)||
| Numerical data| [numerizer](https://github.com/jaidevd/numerizer)|Parse natural language number|
| Data Validation| [pandera](https://github.com/pandera-dev/pandera), [pandas-profiling](https://github.com/pandas-profiling/pandas-profiling)|Pandas|
| Data Cleaning| [pyjanitor](https://github.com/ericmjl/pyjanitor)|Janitor ported to python|
| Parsing| [pyparsing](https://pyparsing-docs.readthedocs.io/en/latest/index.html), [parse](https://pypi.org/project/parse/)||
| Natural date parser| [dateparser](https://github.com/scrapinghub/dateparser)||
| Unicode| [text-unidecode](https://pypi.org/project/text-unidecode/)||
| Emoji| [emoji](https://pypi.org/project/emoji/)||
| Weak Supervision| [snorkel](https://www.snorkel.org/get-started/)||
| Graph Sampling| [little ball of fur](https://github.com/benedekrozemberczki/littleballoffur)||

### Data Exploration  

|Category|Tool|Remarks|
|---|---|---|
|Notebook Tools| [nbdime](https://github.com/jupyter/nbdime)|View Jupyter notebooks through CLI|
|| [papermill](https://github.com/nteract/papermill)| Parametrize notebooks|
| | [nbformat](https://nbformat.readthedocs.io/en/latest/api.html)|Access notebooks programatically|
| | [nbconvert](https://nbconvert.readthedocs.io/en/latest/)|Convert notebooks to other formats|
|Jupyter Extensions| [ipyleaflet](https://github.com/jupyter-widgets/ipyleaflet)|Maps in notebooks|

## Phase: Feature Engineering
### Feature Generation  

|Category|Tool|Remarks|
|---|---|---|
| Automatic feature engineering| [featuretools](https://github.com/FeatureLabs/featuretools), [autopandas](https://autopandas.io/)||
|| [tsfresh](https://github.com/blue-yonder/tsfresh)|Automatic feature engineering for time series|
| Metric learning| [metric-learn](http://contrib.scikit-learn.org/metric-learn/getting_started.html), [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning)||
| Time series| [python-holidays](https://github.com/dr-prodigy/python-holidays)|List of holidays|
| | [skits](https://github.com/ethanrosenthal/skits)|Transformation for time-series data|
| | [catch22](https://github.com/chlubba/catch22)|Pre-built features for time-series data|
| DAG based dataset generation| [DFFML](https://intel.github.io/dffml/usage/integration.html)||

### Dimensionality reduction  

|Category|Tool|Remarks|
|---|---|---|
| Dimensionality reduction| [fbpca](https://github.com/facebook/fbpca), [fitsne](https://github.com/KlugerLab/FIt-SNE)||

## Phase: Modeling

### Model Selection  

|Category|Tool|Remarks|
|---|---|---|
| Find SOTA models| [sotawhat](https://sotawhat.herokuapp.com), [papers-with-code](https://paperswithcode.com/sota)||
||[bert-related-papers](https://github.com/tomohideshibata/BERT-related-papers)|BERT Papers|
||[acl-explorer](http://acl-explorer.eu-west-2.elasticbeanstalk.com/)|ACL Publications Explorer|
| Pretrained models| [modeldepot](https://modeldepot.io/browse), [pytorch-hub](https://pytorch.org/hub)|General|
||[pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch)|Pre-trained ConvNets|
| | [huggingface-models](https://huggingface.co/models)|Transformer Models|
| AutoML| [auto-sklearn](https://automl.github.io/auto-sklearn), [mljar-supervised](https://github.com/mljar/mljar-supervised), [automl-gs](https://github.com/minimaxir/automl-gs)||
||[lazypredict](https://github.com/shankarpandala/lazypredict)|Run all sklearn models at once|
||[tpot](https://github.com/EpistasisLab/tpot)|Genetic AutoML|
||[autocat](https://autocat.apps.allenai.org/)|Auto-generate text classification models in spacy|
| Autogenerate ML code| [mindsdb](https://github.com/mindsdb/mindsdb), [lugwig](https://uber.github.io/ludwig/)||
| ML from command line (or Python or HTTP)| [DFFML](https://intel.github.io/dffml/)||
| Gradient Boosting| [catboost](https://catboost.ai/docs/concepts/about.html), [ngboost](https://github.com/stanfordmlgroup/ngboost)||
||[lightgbm](https://github.com/Microsoft/LightGBM), [thunderbm](https://github.com/Xtra-Computing/thundergbm)|GPU Capable|
| Hidden Markov Models| [hmmlearn](https://github.com/hmmlearn/hmmlearn)||
| Genetic Programming| [gplearn](https://gplearn.readthedocs.io/en/stable/index.html)||
| Active Learning| [modal](https://github.com/modAL-python/modAL)||
| Support Vector Machines| [thundersvm](https://github.com/Xtra-Computing/thundersvm)|Run SVM on GPU|
| Rule based classifier| [sklearn-expertsys](https://github.com/tmadl/sklearn-expertsys)||
| Probabilistic modeling| [pomegranate](https://github.com/jmschrei/pomegranate), [pymc3](https://docs.pymc.io/)||
| Graph Embedding and Community Detection| [karateclub](https://github.com/benedekrozemberczki/karateclub), [python-louvain](https://python-louvain.readthedocs.io/en/latest/)||
| Anomaly detection| [adtk](https://arundo-adtk.readthedocs-hosted.com/en/stable/install.html)||
| Spiking Neural Network| [norse](https://github.com/norse/norse)||
| Fuzzy Learning| [fylearn](https://github.com/sorend/fylearn), [scikit-fuzzy](https://github.com/scikit-fuzzy/scikit-fuzzy)||
| Noisy Label Learning| [cleanlab](https://github.com/cgnorthcutt/cleanlab)||
| Few Shot Learning| [keras-fewshotlearning](https://github.com/ClementWalter/Keras-FewShotLearning)||
| Deep Clustering| [deep-clustering-toolbox](https://github.com/jizongFox/deep-clustering-toolbox)||
| Graph Neural Networks| [spektral](https://github.com/danielegrattarola/spektral/)|GNN for Keras|
| Contrastive Learning| [contrastive-learner](https://github.com/lucidrains/contrastive-learner)||
|Gradient Free Optimization|[nevergrad](https://github.com/facebookresearch/nevergrad)|

### Natural Language Processing  

|Category|Tool|Remarks|
|---|---|---|
| Libraries| [spacy](https://spacy.io/) , [nltk](https://github.com/nltk/nltk), [corenlp](https://stanfordnlp.github.io/CoreNLP/), [deeppavlov](http://docs.deeppavlov.ai/en/master/index.html), [kashgari](https://kashgari.bmio.net/), [camphr (spacy plugin for transformers, elmo, udify)](https://github.com/PKSHATechnology-Research/camphr/), [transformers](https://github.com/huggingface/transformers), [ernie](https://github.com/brunneis/ernie), [stanza](https://stanfordnlp.github.io/stanza/)||
| | [headliner](https://github.com/as-ideas/headliner)|Sequence to sequence models|
| Wrappers| [fast-bert](https://github.com/kaushaltrivedi/fast-bert), [simpletransformers](https://github.com/ThilinaRajapakse/simpletransformers)||
| | [finetune](https://github.com/IndicoDataSolutions/finetune)|Scikit-learn like API for transformers|
| Preprocessing| [textacy](https://github.com/chartbeat-labs/textacy)||
|| [JamSpell](https://github.com/bakwc/JamSpell), [pyhunspell](https://github.com/blatinier/pyhunspell), [pyspellchecker](https://github.com/barrust/pyspellchecker), [cython_hunspell](https://github.com/MSeal/cython_hunspell), [hunspell-dictionaries](https://github.com/wooorm/dictionaries), [autocorrect (can add more languages)](https://github.com/phatpiglet/autocorrect), [symspellpy](https://github.com/mammothb/symspellpy), [spello (train your own spelling correction)](https://github.com/hellohaptik/spello)|Spelling Correction|
| | [contractions](https://github.com/kootenpv/contractions), [pycontractions](https://pypi.org/project/pycontractions/)|Contraction Mapping|
| | [truecase](https://pypi.org/project/truecase/)|Fix casing|
|| [nnsplit](https://github.com/bminixhofer/nnsplit)|Sentence Segmentation|
| | [stopwords-iso](https://github.com/stopwords-iso/stopwords-iso)|Stopwords for all languages|
| | [language-check](https://github.com/myint/language-check)|Language Detection|
| | [neuralcoref](https://github.com/huggingface/neuralcoref)|Coreference Resolution|
| | [inflect](https://pypi.org/project/inflect/)|Inflections|
| | [scrubadub](https://scrubadub.readthedocs.io/en/stable/#)|PID removal|
| Tokenization| [sentencepiece](https://github.com/google/sentencepiece), [youtokentome](https://github.com/VKCOM/YouTokenToMe), [subword-nmt](https://github.com/rsennrich/subword-nmt)||
| | [jieba](https://github.com/fxsjy/jieba)|Chinese Word Segmentation|
| Embeddings| [InferSent](https://github.com/facebookresearch/InferSent), [bert-as-service](https://github.com/hanxiao/bert-as-service), [sent2vec](https://github.com/NewKnowledge/nk-sent2vec), [sense2vec](https://github.com/explosion/sense2vec), [BM25Transformer](https://github.com/arosh/BM25Transformer),[glove-python](https://github.com/maciejkula/glove-python), [fse](https://github.com/oborchers/Fast_Sentence_Embeddings)||
|| [sentence-transformers](https://github.com/UKPLab/sentence-transformers)|BERT sentence embeddings|
||[pymagnitude](https://github.com/plasticityai/magnitude)|Access word-embeddings programatically|
||[chakin](https://github.com/chakki-works/chakin)|Download pre-trained word vectors|
||[zeugma](https://github.com/nkthiebaut/zeugma)|Pretrained-word embeddings as scikit-learn transformers|
||[starspace](https://github.com/facebookresearch/StarSpace)|Learn embeddings for anything|
| Cross-lingual Embeddings| [muse](https://github.com/facebookresearch/MUSE), [laserembeddings](https://pypi.org/project/laserembeddings/), [xlm](https://github.com/facebookresearch/XLM), [LaBSE](https://tfhub.dev/google/LaBSE/1)||
||[BPEmb](https://nlp.h-its.org/bpemb/)|Subword Embeddings in 275 Languages|
| Multilingual support| [polyglot](https://polyglot.readthedocs.io/en/latest/index.html)||
|| [inltk](https://github.com/goru001/inltk), [indic_nlp](https://github.com/anoopkunchukuttan/indic_nlp_library)|Indic Languages|
| Knowledge| [conceptnet-lite](https://github.com/ldtoolkit/conceptnet-lite)||
| | [stanford-openie](https://github.com/philipperemy/Stanford-OpenIE-Python)|Knowledge Graphs|
| Domain-specific BERT| [codebert](https://huggingface.co/codistai/codeBERT-small-v2#)|Code|
|| [clinicalbert-mimicnotes](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT), [clinicalbert-discharge-summary](https://huggingface.co/emilyalsentzer/Bio_Discharge_Summary_BERT)|Clinical Domain|
|Scientific Domain| [scispacy](https://github.com/allenai/scispacy)|Spacy for bio-medical data|
| Text Extraction| [textract (Image, Audio, PDF)](https://textract.readthedocs.io/en/stable/)||
| Text Generation| [gp2client](https://github.com/rish-16/gpt2client), [textgenrnn](https://github.com/minimaxir/textgenrnn), [gpt-2-simple](https://github.com/minimaxir/gpt-2-simple), [aitextgen](https://github.com/minimaxir/aitextgen)|GPT-2|
| | [markovify](https://github.com/jsvine/markovify)|Markov chains|
| Machine Translation| [MarianMT](https://huggingface.co/transformers/model_doc/marian.html)||
| | [googletrans](https://pypi.org/project/googletrans/), [word2word](https://github.com/Kyubyong/word2word), [translate-python](https://github.com/terryyin/translate-python)|Translation libraries|
| Summarization| [textrank](https://github.com/summanlp/textrank), [pytldr](https://github.com/jaijuneja/PyTLDR), [bert-extractive-summarizer](https://github.com/dmmiller612/bert-extractive-summarizer), [sumy](https://github.com/miso-belica/sumy), [fast-pagerank](https://github.com/asajadi/fast-pagerank), [sumeval](https://github.com/chakki-works/sumeval)||
| Keyword extraction| [rake](https://github.com/zelandiya/RAKE-tutorial), [pke](https://github.com/boudinfl/pke), [phrasemachine](https://github.com/slanglab/phrasemachine)||
| Multiply Choice Question Answering| [mcQA](https://github.com/mcQA-suite/mcQA)||
|Ranking|[transformer-rankers](https://github.com/Guzpenha/transformer_rankers)||
| NLU| [snips-nlu](https://github.com/snipsco/snips-nlu)||
| Semantic parsing| [quepy](https://github.com/machinalis/quepy)||
| Readability| [homer](https://github.com/wyounas/homer)||
| Topic Modeling| [guidedlda](https://github.com/vi3k6i5/guidedlda), [enstop](https://github.com/lmcinnes/enstop), [top2vec](https://github.com/ddangelov/Top2Vec), [contextualized-topic-models](https://github.com/MilaNLProc/contextualized-topic-models), [corex_topic](https://github.com/gregversteeg/corex_topic), [lda2vec](https://github.com/cemoody/lda2vec)||
| Clustering| [kmodes](https://github.com/nicodv/kmodes), [star-clustering](https://github.com/josephius/star-clustering)||
||[spherecluster](https://github.com/jasonlaska/spherecluster)|K-means with cosine distance|
||[kneed](https://github.com/arvkevi/kneed)|Automatically find number of clusters from elbow curve|
| Metrics| [seqeval](https://github.com/chakki-works/seqeval)|NER, POS tagging|
| String match|[phrase-seeker](https://github.com/kirillgashkov/phrase-seeker), [textsearch](https://github.com/kootenpv/textsearch)||
||[jellyfish](https://pypi.org/project/jellyfish/)|Perform string and phonetic comparison|
||[flashtext](https://github.com/vi3k6i5/flashtext)|Super-fast extract and replace keywords|
||[pythonverbalexpressions](https://github.com/VerbalExpressions/PythonVerbalExpressions)|Verbally describe regex|
||[commonregex](https://github.com/madisonmay/CommonRegex)|Ready-made regex for email/phone etc.|
| | [textdistance](https://github.com/life4/textdistance), [editdistance](https://github.com/aflc/editdistance), [word-mover-distance](https://radimrehurek.com/gensim/models/keyedvectors.html#what-can-i-do-with-word-vectors)|Text distances|
||[wmd-relax](https://github.com/src-d/wmd-relax)|Word mover distance for spacy|
|  | [fuzzywuzzy](https://github.com/seatgeek/fuzzywuzzy), [spaczz](https://github.com/gandersen101/spaczz)|Fuzzy Search|
| Sentiment| [vaderSentiment](https://github.com/cjhutto/vaderSentiment)|Rule based|
| | [absa](https://github.com/ScalaConsultants/Aspect-Based-Sentiment-Analysis)|Aspect Based Sentiment Analysis|
| Emotion Classification| [distilroberta-finetuned](https://huggingface.co/mrm8488/distilroberta-base-finetuned-sentiment), [goemotion-pytorch](https://github.com/monologg/GoEmotions-pytorch)||
| Profanity detection| [profanity-check](https://github.com/vzhou842/profanity-check)||
| Visualization| [stylecloud](https://github.com/minimaxir/stylecloud)|Word Clouds|
| | [scattertext](https://github.com/JasonKessler/scattertext)|Compare word usage across segments|
| Named Entity Recognition(NER) | [spaCy](https://spacy.io/) , [Stanford NER](https://nlp.stanford.edu/software/CRF-NER.shtml), [sklearn-crfsuite](https://sklearn-crfsuite.readthedocs.io/en/latest/index.html)||
||[med7](https://github.com/kormilitzin/med7)|Spacy NER for medical records|
| Fill blanks| [fitbert](https://github.com/Qordobacode/fitbert)||
| Dictionary| [vocabulary](https://vocabulary.readthedocs.io/en/latest/usage.html)||
| Nearest neighbor| [faiss](https://github.com/facebookresearch/faiss)||
| Knowledge Distillation| [textbrewer](https://github.com/airaria/TextBrewer), [aquvitae](https://github.com/aquvitae/aquvitae)||
| Sentence Coherence| [lm-scorer](https://github.com/simonepri/lm-scorer), [bertscore](https://github.com/Tiiiger/bert_score)||
| Record Linking| [fuzzymatcher](https://github.com/RobinL/fuzzymatcher)||


### Computer Vision  

|Category|Tool|Remarks|
|---|---|---|
| Pretrained models| [pytorchcv](https://pypi.org/project/pytorchcv/)||
| Image processing| [scikit-image](https://github.com/scikit-image/scikit-image), [imutils](https://github.com/jrosebr1/imutils)||
| Segmentation Models| [segmentation_models](https://github.com/qubvel/segmentation_models)|Keras|
| Face recognition| [face_recognition](https://github.com/ageitgey/face_recognition)||
||[face-alignment](https://github.com/1adrianb/face-alignment)|Find facial landmarks|
| GANS| [mimicry](https://mimicry.readthedocs.io/en/latest/index.html)||
| Face swapping| [faceit](https://github.com/goberoi/faceit), [faceit-live](https://github.com/alew3/faceit_live), [avatarify](https://github.com/alievk/avatarify)||
| Video summarization| [videodigest](https://github.com/agermanidis/videodigest)||
| Semantic search over videos| [scoper](https://github.com/RameshAditya/scoper)||
| OCR| [keras-ocr](https://github.com/faustomorales/keras-ocr), [pytesseract](https://github.com/madmaze/pytesseract)||
||[easyocr](https://github.com/JaidedAI/EasyOCR)|40+ languages|
| Object detection| [luminoth](https://github.com/tryolabs/luminoth), [detectron2](https://github.com/facebookresearch/detectron2), [mmdetection](https://github.com/open-mmlab/mmdetection)||
| Image hashing| [ImageHash](https://pypi.org/project/ImageHash/)||

### Audio  

|Category|Tool|Remarks|
|---|---|---|
| Library| [speech_recognition](https://github.com/Uberi/speech_recognition), [pyannotate](http://pyannote.github.io/), [librosa](https://librosa.github.io/librosa/index.html)||
| Diarization| [resemblyzer](https://github.com/resemble-ai/Resemblyzer)||
| Source Separation| [spleeter](https://github.com/deezer/spleeter), [nussl](https://github.com/nussl/nussl), [open-unmix-pytorch](https://github.com/sigsep/open-unmix-pytorch), [asteroid](https://github.com/mpariente/asteroid)||

### Recommendation System  

|Category|Tool|Remarks|
|---|---|---|
|Libraries| [xlearn](https://github.com/aksnzhy/xlearn), [DeepCTR](https://github.com/shenweichen/DeepCTR)| Factorization machines (FM), and field-aware factorization machines (FFM)|
||[lightfm](https://github.com/lyst/lightfm), [spotlight](https://github.com/maciejkula/spotlight)|Popular Recsys algos|
| Collaborative Filtering| [implicit](https://github.com/benfred/implicit)||
| Scikit-learn like API| [surprise](https://github.com/NicolasHug/Surprise)||
| Recommendation System in Pytorch| [CaseRecommender](https://github.com/caserec/CaseRecommender)||
| Apriori algorithm| [apyori](https://github.com/ymoch/apyori)||
|Metrics|[rs_metrics](https://darel13712.github.io/rs_metrics/metrics/)||

### Timeseries  

|Category|Tool|Remarks|
|---|---|---|
| Libraries| [prophet](https://facebook.github.io/prophet/docs/quick_start.html#python-api), [tslearn](https://github.com/tslearn-team/tslearn), [pyts](https://github.com/johannfaouzi/pyts), [seglearn](https://github.com/dmbee/seglearn), [cesium](https://github.com/cesium-ml/cesium), [stumpy](https://github.com/TDAmeritrade/stumpy), [darts](https://github.com/unit8co/darts)||
|| [sktime](https://github.com/alan-turing-institute/sktime)|Scikit-learn like API|
||[atspy](https://github.com/firmai/atspy)|Automated time-series models|
| ARIMA models| [pmdarima](https://github.com/alkaline-ml/pmdarima)||

### Framework extensions  

|Category|Tool|Remarks|
|---|---|---|
|Addons| [mlxtend](https://github.com/rasbt/mlxtend)|Extra utilities not present in frameworks|
| Pytorch| [pytorch-summary](https://github.com/sksq96/pytorch-summary)|Keras-like summary|
||[skorch](https://github.com/skorch-dev/skorch)|Wrap pytorch in scikit-learn compatible API|
||[pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)|Lightweight wrapper for PyTorch|
||[einops](https://github.com/arogozhnikov/einops)|Einstein Notation|
||[kornia](https://torchgeometry.readthedocs.io/en/latest/index.html)|Computer Vision Methods|
||[torchcontrib](https://github.com/pytorch/contrib)|SOTA Bulding Blocks in PyTorch|
||[pytorch-optimizer](https://github.com/jettify/pytorch-optimizer)|Collection of optimizers|
| Scikit-learn| [scikit-lego](https://scikit-lego.readthedocs.io/en/latest/index.html), [iterative-stratification](https://github.com/trent-b/iterative-stratification)||
||[tscv](https://github.com/WenjieZ/TSCV)|Time-series cross-validation|
||[iterstrat](https://github.com/trent-b/iterative-stratification)|Cross-validation for multi-label data|
| Keras| [tf-sha-rnn](https://github.com/titu1994/tf-sha-rnn)||
||[keras-radam](https://github.com/CyberZHG/keras-radam)|RADAM optimizer|
||[larq](https://github.com/larq/larq)|Binarized neural networks|
||[ktrain](https://pypi.org/project/ktrain/)|FastAI like interface for keras|
||[tavolo](https://github.com/eliorc/tavolo)|Kaggle Tricks as Keras Layers|
| Tensorflow| [tensorflow-addons](https://github.com/tensorflow/addons)||

## Phase: Validation
### Model Training Monitoring  

|Category|Tool|Remarks|
|---|---|---|
| Learning curve| [lrcurve](https://github.com/AndreasMadsen/python-lrcurve), [livelossplot](https://github.com/stared/livelossplot)|Plot realtime learning curve in Keras|
| Notification| [knockknock](https://github.com/huggingface/knockknock)|Get notified by slack/email|
| | [jupyter-notify](https://github.com/ShopRunner/jupyter-notify)|Notify when task is completed in jupyter|
| Progress bar| [fastprogress](https://github.com/fastai/fastprogress), [tqdm](https://github.com/tqdm/tqdm)||
| GPU Usage| [gpumonitor](https://github.com/sicara/gpumonitor)||
| | [jupyterlab-nvdashboard](https://github.com/rapidsai/jupyterlab-nvdashboard)|See GPU Usage in jupyterlab|

### Interpretability  

|Category|Tool|Remarks|
|---|---|---|
| Interpret models| [eli5](https://eli5.readthedocs.io/en/latest/), [lime](https://github.com/marcotcr/lime), [shap](https://github.com/slundberg/shap), [alibi](https://github.com/SeldonIO/alibi), [tf-explain](https://github.com/sicara/tf-explain), [treeinterpreter](https://github.com/andosa/treeinterpreter), [pybreakdown](https://github.com/MI2DataLab/pyBreakDown), [xai](https://github.com/EthicalML/xai), [lofo-importance](https://github.com/aerdem4/lofo-importance), [interpretML](https://github.com/interpretml/interpret)||
| Interpret BERT| [exbert](http://exbert.net/exBERT.html?sentence=I%20liked%20the%20music&layer=0&heads=..0,1,2,3,4,5,6,7,8,9,10,11&threshold=0.7&tokenInd=null&tokenSide=null&maskInds=..9&metaMatch=pos&metaMax=pos&displayInspector=null&offsetIdxs=..-1,0,1&hideClsSep=true)||
||[bertviz](https://github.com/jessevig/bertviz)|Explore self-attention in BERT|
| Interpret word2vec| [word2viz](https://lamyiowce.github.io/word2viz/), [whatlies](https://github.com/RasaHQ/whatlies)||

### Visualization  

|Category|Tool|Remarks|
|---|---|---|
| Libraries| [pygal](http://www.pygal.org/en/latest/index.html), [plotly](https://github.com/plotly/plotly.py), [plotnine](https://github.com/has2k1/plotnine)||
|| [yellowbrick](https://www.scikit-yb.org/en/latest/index.html), [scikit-plot](https://scikit-plot.readthedocs.io/en/stable/metrics.html)|Visualization for scikit-learn|
|| [pyldavis](https://pyldavis.readthedocs.io/en/latest/)|Visualize topics models|
| Interactive charts| [bokeh](https://github.com/bokeh/bokeh)||
|| [flourish-studio](https://flourish.studio/)|Create interactive charts online|
|| [mpld3](http://mpld3.github.io/index.html)|Matplotlib to D3 Converter|
|Model Visualization|[netron](https://github.com/lutzroeder/netron), [nn-svg](http://alexlenail.me/NN-SVG/LeNet.html)|Architecture|
|| [keract](https://github.com/philipperemy/keract)|Activation maps for keras|
|| [keras-vis](https://github.com/raghakot/keras-vis)|Visualize keras models|
| Styling| [open-color](https://yeun.github.io/open-color/)|Color Schemes|
||[mplcyberpunk](https://github.com/dhaitz/mplcyberpunk)|Cyberpunk style for matplotlib|
|| [chart.xkcd](https://timqian.com/chart.xkcd/)|XKCD like charts|
| Generate graphs using markdown| [mermaid](https://mermaid-js.github.io/mermaid/#/README)||
| High dimensional visualization| [umap](https://github.com/lmcinnes/umap)||
||[ivis](https://github.com/beringresearch/ivis)|Ivis Algorithm|
| Bar chart race animation| [bar_chart_race](https://github.com/dexplo/bar_chart_race)||

## Phase: Optimization
### Hyperparameter Optimization  

|Category|Tool|Remarks|
|---|---|---|
| General| [hyperopt](https://github.com/hyperopt/hyperopt), [optuna](https://optuna.org/), [evol](https://github.com/godatadriven/evol), [talos](https://github.com/autonomio/talos)||
| Keras| [keras-tuner](https://github.com/keras-team/keras-tuner)||
| Scikit-learn|[hyperopt-sklearn](https://github.com/hyperopt/hyperopt-sklearn)|Bayesian Optimization|
||[sklearn-deap](https://github.com/rsteca/sklearn-deap)|Evolutionary algorithm|
| Parameter optimization| [ParameterImportance](https://github.com/automl/ParameterImportance)||

## Phase: Production
### Model Serialization  

|Category|Tool|Remarks|
|---|---|---|
| Transpiling| [sklearn-porter](https://github.com/nok/sklearn-porter), [m2cgen](https://github.com/BayesWitnesses/m2cgen)|Transpile sklearn model to C, Java, JavaScript and others|
||[hummingbird](https://github.com/microsoft/hummingbird)|Convert ML models to PyTorch|
| Pickling extended| [cloudpickle](https://github.com/cloudpipe/cloudpickle), [jsonpickle](https://github.com/jsonpickle/jsonpickle)||

### Scalability  

|Category|Tool|Remarks|
|---|---|---|
| Parallelize Pandas| [pandarallel](https://github.com/nalepae/pandarallel), [swifter](https://github.com/jmcarpenter2/swifter), [modin](https://github.com/modin-project/modin)||
|Pandas on Huge data|[vaex](https://github.com/vaexio/vaex)||
| Parallelize numpy operations| [numba](http://numba.pydata.org/)||
| Distributed training| [horovod](https://github.com/horovod/horovod)||

### Bechmark  

|Category|Tool|Remarks|
|---|---|---|
| Profile pytorch layers| [torchprof](https://github.com/awwong1/torchprof)||
| Load testing| [k6](https://k6.io/)||
| Monitor GPU usage| [nvtop](https://github.com/Syllo/nvtop)||

### API  

|Category|Tool|Remarks|
|---|---|---|
|API Frameworks|[flask](https://flask.palletsprojects.com/en/1.1.x/)||
||[fastapi](https://fastapi.tiangolo.com/)|Automatic Docs and Validation|
| Configuration Management| [config](https://pypi.org/project/config/), [python-decouple](https://github.com/henriquebastos/python-decouple)||
| Data Validation| [schema](https://github.com/keleshev/schema), [jsonschema](https://pypi.org/project/jsonschema/), [cerebrus](https://github.com/pyeve/cerberus), [pydantic](https://pydantic-docs.helpmanual.io/), [marshmallow](https://marshmallow.readthedocs.io/en/stable/), [validators](https://validators.readthedocs.io/en/latest/#basic-validators)||
| Enable CORS in Flask| [flask-cors](https://flask-cors.readthedocs.io/en/latest/)||
| Caching| [cachetools](https://pypi.org/project/cachetools/), [cachew (cache to local sqlite)](https://github.com/karlicoss/cachew)||
| Authentication| [pyjwt (JWT)](https://github.com/jpadilla/pyjwt)||
| Task Queue| [rq](https://github.com/rq/rq), [schedule](https://github.com/dbader/schedule), [huey](https://github.com/coleifer/huey)||
| Database| [flask-sqlalchemy](https://github.com/pallets/flask-sqlalchemy), [tinydb](https://github.com/msiemens/tinydb), [flask-pymongo](https://flask-pymongo.readthedocs.io/en/latest/)||
| Logging| [loguru](https://github.com/Delgan/loguru)||

### Dashboard  

|Category|Tool|Remarks|
|---|---|---|
|Libraries| [streamlit](https://github.com/streamlit/streamlit)|Generate frontend with python|
||[gradio](https://github.com/gradio-app/gradio)|Fast UI generation for prototyping|
||[dash](https://plotly.com/dash/)|React Dashboard using Python|
||[voila](https://github.com/voila-dashboards/voila)|Convert Jupyter notebooks into dashboard|

### Adversarial testing  

|Category|Tool|Remarks|
|---|---|---|
| Generate images to fool model| [foolbox](https://github.com/bethgelab/foolbox)||
| Generate phrases to fool NLP models| [triggers](https://www.ericswallace.com/triggers)||
| General| [cleverhans](https://github.com/tensorflow/cleverhans)||

### Python libraries  

|Category|Tool|Remarks|
|---|---|---|
| Decorators| [retrying (retry some function)](https://pypi.org/project/retrying/)||
| bloom filter| [python-bloomfilter](https://github.com/jaybaird/python-bloomfilter)||
| Run python libraries in sandbox| [pipx](https://github.com/pipxproject/pipx)||
| Pretty print tables in CLI| [tabulate](https://pypi.org/project/tabulate/)||
| Leaflet maps from python| [folium](https://python-visualization.github.io/folium/)||
| Debugging| [PySnooper](https://github.com/cool-RR/PySnooper)||
| Date and Time| [pendulum](https://github.com/sdispater/pendulum)||
| Create interactive prompts| [prompt-toolkit](https://pypi.org/project/prompt-toolkit/)||
| Concurrent database| [pickleshare](https://pypi.org/project/pickleshare/)||
| Aync| [tomorrow](https://github.com/madisonmay/Tomorrow)||
| Testing| [crosshair(find failure cases for functions)](https://github.com/pschanely/CrossHair)||
| Virtual webcam| [pyfakewebcam](https://github.com/jremmons/pyfakewebcam)||
| CLI Formatting| [rich](https://github.com/willmcgugan/rich)||
| Control mouse and output device| [pynput](https://pypi.org/project/pynput/)||
| Shell commands as functions| [sh](http://amoffat.github.io/sh/)||
|Standard Library Extension|[ubelt](https://github.com/Erotemic/ubelt)||
|Improved doctest|[xdoctest](https://github.com/Erotemic/xdoctest)||

### Workflow  

|Category|Tool|Remarks|
|---|---|---|
|Linux|[ripgrep](https://github.com/phiresky/ripgrep-all)||
|Colab| [colab-cli](https://github.com/Akshay090/colab-cli) | Manager colab notebook from command line|
|Git|[gitjk](https://github.com/mapmeld/gitjk)|Undo what you just did in git|
