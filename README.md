# Text2KG

A collection of models for the paper "Seq2KG: An End-to-End Neural Model for Domain Agnostic Knowledge Graph (not Text Graph) Construction from Text" by Michael Stewart and Wei Liu.

## File structure
    
    datasets // Contains all three datasets mentioned in the paper
    sourcecode
        candidate_extraction    // The rule-based model used as a baseline in the paper.
        candidate_filtering     // The neural network-based filtering model
        end_to_end_model        // A GRU-based model to predict all triples in a given document
        joint_model             // The Seq2KG model - same as above, but also predicts entity types
        evaluation              // Scripts for evaluating the performance of each model
        redcoat_parser          // Scripts to convert Redcoat annotation files into datasets for filtering and end_to_end models
        requirements.txt        // All Python requirements for the /sourcecode directory
        

## Requirements

The code in this repository was written using Python 3.6. You can install the required Python libraries via `python -m pip install sourcecode/requirements.txt`. We recommend setting up a virtual environment prior to installing packages.

Prior to running any of the models, you must first install the Bert-Base, Cased BERT model and place it under `sourcecode/bert`. The model is available [here](https://github.com/google-research/bert).

Once the model has been placed under `sourcecode/bert/cased_L-12_H-768_A-12`, run `bert-as-a-service` to allow our models to interact with the BERT model and obtain embeddings:

    $ cd sourecode/bert
    $ ./run_server_e2e.sh
    $ ./run_server_filter.sh

## Running the models

### Candidate extraction (i.e. the Rule-based model)

To run the rule-based model, simply navigate to `sourcecode/candidate_extraction` and run:

    $ python triples_from_test_data.py

This will process the datasets under `datasets` and output a list of triples per document to `sourcecode/candidate_extraction/output`.

### Candidate filtering (i.e. the Filtering model)

To run the filtering model, navigate to `sourcecode/candidate_filtering`. First, build the dataset with:

    $ python build_data.py <dataset>

where `<dataset>` is either `bbn`, `automotiveEngineering`, or `cateringServices`. Then, train the model via:

    $ python train.py <dataset>

Once the model has been trained, it can be evaluated on the test data under `datasets/` via:

    $ python triples_from_test_data.py

### End to end and Joint models (Seq2KG)

Training and evaluating the end to end and joint models is exactly the same as the candidate filtering model above. The end-to-end model performs only triple extraction, while the joint model also identifies the entity type(s) of each extracted head and tail in each triple.

## Evaluating the models

There is an evaluation script under `sourcecode/evaluation` that can be used to evaluate the performance across all models:

    $ cd sourcecode/evaluation
    $ python evaluator.py

The evaluation results will be printed to the terminal window.

## Annotated datasets

As part of our submission we have included three manually-annotated datasets, available under `datasets/`. Two of the datasets were obtained by scraping Carbuzz.com (automotiveEngineering) and Eater.com (cateringServices). The BBN dataset is a standard benchmarking entity typing dataset, obtained [here](https://github.com/INK-USC/AFET/tree/master/Data/BBN), but also labelled with triples.

Each of the three datasets were labelled for triple extraction using our annotation tool, Redcoat (http://agent.csse.uwa.edu.au/redcoat/). The datasets are labelled in a similar manner to entity typing datasets. An example document:

    {"doc_idx": 0, "tokens": ["Barrack", "spoke", "to", "Michelle", "at", "the", "Whitehouse", "."], "mentions": [{"start": 0, "end": 1, "labels": ["head", "head/1"], ... } ... }, ...

More details about the datasets are available in the paper.