## Introduction

NLP project at ENSAE Paris under the supervizion of Benjamin Muller.

The goal of the project is to build a conditional GPT-2 to generate fake news. 
The [NewsAggregator dataset](https://archive.ics.uci.edu/ml/datasets/News+Aggregator) is used for this project. 
News and keywords were extracted from the dataset using python scripts from https://github.com/ivanlai/Conditional_Text_Generation.

The Colab notebook to reproduce our results is available here : https://colab.research.google.com/drive/1dzOrDhmXu-s3BVxdI2P6UgHBCsjRVxOO?usp=sharing


## Training

To download, setup the project and install the requirements, please run the following command lines :
```shell script
$ git clone "https://gitlab.com/matthieu_futeral/fake_news_generation.git"
$ cd fake_news_generation
$ pip install -r requirements.txt
```

To download the processed data :
```shell script
$ wget "https://www.dropbox.com/s/k3uw307myypkmba/data.tar.gz"
$ tar -xzvf data.tar.gz
$ rm data.tar.gz
```

Fine-tuning light GPT-2 (from huggingface) :
```shell script
$ python3 training.py --epochs 4 --batch_size 2 --lr 2e-4 --gradient_step 32

    options:
          --epochs (int) : number of epochs for fine-tuning GPT-2 (default: 100)
          --batch_size (int) : size of the batch (default: 32)
          --lr (float) : learning rate (default: 1e-4)
          --gradient_step (int) : number of steps before gradient update to overcome RAM issues (default: 16)
```


## Generate Conditional Fake News

To generate news from a fine-tune GPT-2 (please make sure you have fine-tune a GPT-2 before this step) :
```shell script
$ python3 generator.py --n_sentences 5 --topk 10 --temperature 0.7  --cat e  --title "Some title"  \
                       --keywords keyword1 keyword2  --beam_search 0

    options:
          --n_sentences (int) : number of news to generate (default: 10)
          --topk (int) : If positive, generates according to topk method, selecting topk predictions to sample from (default: 0)
          --temperature (float) : temperature if using topk method (default: 0.0)
          --cat (str) : category of the news to generate (Required)
          --tilte (str) : Title of the news (Optional)
          --keywords (str) : keywords of the news (Optional)
          --beam_search (int) : If positive, generates according to beam search, arg is the length of the beam (default: 0)
```


## Analysis

To run Latent Semantic Analysis :
```shell script
$ python3 latent_semantic_analysis.py --n_news 500 --n_components 256

    options:
          --n_news (int) : number of news randomly selected from the dataset (default: 500)
          --n_components (int) : number of components in the SVD (default: 256)
          --generated (store_true) : perform the analysis on the generated data
```

To run the BERT classifier :
```shell script
$ python3 classifier.py --epochs 30 --batch_size 4 --lr 1e-4 --gradient_step 8 --training --patience 5

    options:
          --epochs (int) : number of epochs for fine-tuning BERT (default: 100)
          --batch_size (int) : size of the batch (default: 32)
          --lr (float) : learning rate (default: 1e-4)
          --gradient_step (int) : number of steps before gradient update to overcome RAM issues (default: 16)
          --patience (int) : number of epochs before early stopping
          --training (store_true) : perform training on the NEwsAggregator dataset else testing on the generated news
```


## References

<a id="1">[1]</a> 
Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya,
Language Models are Unsupervised Multitask Learners,
2019

<a id="2">[2]</a> 
Thomas Wolf et al., 
Transformers: State-of-the-Art Natural Language Processing, 
Association for Computational Linguistics,
2020
 
<a id="3">[3]</a> 
Devlin, Jacob  and Chang, Ming-Wei  and Lee, Kenton  and Toutanova, Kristina,
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,
Association for Computational Linguistics,
2019