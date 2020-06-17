from torch.utils.data import DataLoader
import sys
import numpy as np
import pickle as pkl
from bert_serving.client import BertClient
import codecs, math
import pandas as pd
import os, time
from data_utils import TriplesDataset, save_obj_to_pkl_file, load_documents

from logger import logger

from config import Config
cf = Config()
bc = BertClient(port=6555, port_out=6556)

# Build the given dataset (train or test) given the dataframe and corresponding documents.
def build_dataset(dataframe, ds_name, documents):
	triples = []
	invalid_triples_count = 0
	

	# A row is considered valid if it is not missing any attributes.
	def check_row_is_valid(row):
		return not (pd.isnull(getattr(row, 's1')) or pd.isnull(getattr(row, 'r')) or pd.isnull(getattr(row, 's2')) or pd.isnull(getattr(row, 'label')))

	# 1. Extract the document, head, relation, tail and label from each row in the CSV.
	# Encode it via Bert as a Service.
	logger.info("Encoding triples via Bert as a Service...")
	for i, row in enumerate(dataframe.itertuples()):	
		
		if check_row_is_valid(row):	
			idx = i
			doc_idx = int(getattr(row, 'index'))


			#document = bc.encode([documents[int(getattr(row, 'index'))]])[0]
			document = [0]
			head 	 = bc.encode([getattr(row, 's1')])[0]
			relation = bc.encode([getattr(row, 'r')])[0]
			tail 	 = bc.encode([getattr(row, 's2')])[0]
			label    = getattr(row, 'label')
			triples.append((idx, doc_idx, document, head, relation, tail, label))	
		else:
			invalid_triples_count += 1
		sys.stdout.write("\r%i / %i\t\tInvalid rows: %d" % (i, len(dataframe), invalid_triples_count))
		sys.stdout.flush()

	print("")
	if invalid_triples_count > 0:
		logger.warn("%d triples were invalid and were not added to the %s dataset." % (invalid_triples_count, ds_name))
	logger.info("Building data loader...")
	
	# Construct a dataloader.
	data_idx, data_doc_idx, data_doc, data_head, data_relation, data_tail, data_y = [], [], [], [], [], [], []
	for i, triple in enumerate(triples):
		data_idx.append(np.asarray(triple[0]))
		data_doc_idx.append(np.asarray(triple[1]))
		data_doc.append(np.asarray(triple[2]))
		data_head.append(np.asarray(triple[3]))
		data_relation.append(np.asarray(triple[4]))
		data_tail.append(np.asarray(triple[5]))
		data_y.append(np.asarray(float(triple[6])))

		sys.stdout.write("\r%i / %i" % (i, len(triples)))
		sys.stdout.flush()

	print("")
	logger.info("Data loader complete.")

	dataset = TriplesDataset(data_idx, data_doc_idx, data_doc, data_head, data_relation, data_tail, data_y)
	return dataset




def main(opts):

	if len(opts) == 0:
		raise ValueError("Usage: evaluate.py <dataset>")
	dataset = opts[0]
	if dataset not in ['cateringServices', 'automotiveEngineering', 'bbn']:
		raise ValueError("Dataset must be either cateringServices, automotiveEngineering or bbn.")

	cf.load_config(dataset)

	datasets = { }
	data_loaders = {}

	# 1. Read in the train and dev datasets from the csv file.
	datasets['dev']  = pd.read_csv(cf.DEV_FILENAME)
	datasets['train'] = pd.read_csv(cf.TRAIN_FILENAME, encoding='utf-8')
	

	# 2. Load documents
	documents = {}
	documents['train'] = load_documents(cf.TRAIN_DOCUMENTS_FILENAME)
	documents['dev']   = load_documents(cf.DEV_DOCUMENTS_FILENAME)

	# 3. Build a data loader for each dataset (train, dev, test).
	data_loaders = {}
	for ds_name, dataset in datasets.items():
		logger.info("Building %s dataset..." % (ds_name))
		dataset = build_dataset(dataset, ds_name, documents[ds_name])
		data_loader = DataLoader(dataset, batch_size=cf.BATCH_SIZE, pin_memory=True)
		data_loaders[ds_name] = data_loader
		logger.info("The %s dataset was built successfully." % ds_name)


	logger.info("Saving data loaders to file...")

	save_obj_to_pkl_file(data_loaders['train'], 'data loader (train)', cf.ASSET_FOLDER + '/data_loader_train.pkl')
	save_obj_to_pkl_file(data_loaders['dev'], 'data loader (dev)', cf.ASSET_FOLDER + '/data_loader_dev.pkl')


if __name__ == "__main__":
	main(sys.argv[1:])
