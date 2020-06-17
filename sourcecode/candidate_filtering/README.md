# Candidate Filtering 

This folder contains a Pytorch model that discriminates between relevant and irrelevant triples.

## Requirements

All of the code was built using Python 3.6. It has not been tested on earlier versions of Python.

The following Python packages are required:
- torch 1.0.0+
- tensorflow-gpu 1.12.0+
- colorama
- bert-serving-server
- bert-serving-client

The easiest way to install all required libraries at once is to navigate to the `../sourcecode` folder and run:

   ```
   $ pip install -r requirements.txt
   ```

## Initial setup

1. Before running the code for the first time, you will need to download the [Bert Base, Cased model](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip), extract it, and save the contents of the folder to `/bert/models/cased_L-12_H-768_A-12`. 
2. Install ["Bert as a Service"](https://github.com/hanxiao/bert-as-service) by following the guide on the Github repository. The server component must be installed via Python3, as it does not support earlier versions of Python.
3. Run the Bert as a Service server via the command:

   ```
   $ bert/run_server.sh
   ```

## Model configuration

The model configuration is located in `config.json`. There are a number of options:

	"dataset": "<industry>",           // The dataset, i.e. industry, such as "cateringServices"
	"embedding_dim": 768,              // The embedding dimension. Should be 768 with Bert BASE
	"hidden_dim": 768,                 // The dimension of the hidden layer(s)
	"learning_rate": 0.0001,           // The learning rate
	"max_seq_len": 100,                // Maximum length of any sequence to be encoded by BERT
	"stop_condition": 50,              // Number of epochs to run through before stopping due to no performance improvement
	"batch_size": 10                   // The batch size

## Dataset format

Before running the model for the first time, you'll need to run the dataset creation script under `/sourcecode/redcoat_parser` (if you haven't already):

```
$ cd ../redcoat_parser
$ python create_datasets.py
$ cd ../filtering_model
```

More information about how this script works will be available under `/sourcecode/redcoat_parser/README.md` in future.

## Training the model

First, build the dataset via:

```
$ python build_data.py <industry>
```

This will take the corresponding dataset stored under `dataset_folder` and construct a Pytorch Dataloader for it. During this process, Bert as a Service will embed each of the head, relations, tails, and documents associated with each triple.

The data loaders will be saved under `models/<industry>/asset`, so you can run the same experiment multiple times without needing to rebuild the data loaders every time.

Next, you can train the model:

```
$ python train.py <industry>
```

The script will evaluate the model's performance on the development dataset during training.

## Evaluating the model

The model may be evaluated by calling the `triples_from_contest_data.py` script:

```
$ python triples_from_contest_data.py
```

This will load the model and contest data from `../../datasets/icdm_contest_data.csv` and predict all triples. At present, the only available models are automotiveEngineering and cateringServices so the output file, `output/submission.csv` only contains triples from those industries. These may be manually joined with our best `submission.csv` file if necessary.


