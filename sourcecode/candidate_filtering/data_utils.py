import pickle as pkl 
from torch.utils.data import Dataset
from logger import logger
import torch
import codecs, sys
import numpy as np
from config import device

# Load documents from a given filename.
def load_documents(filename):
	with open(filename, 'r', encoding='utf-8') as f:
		documents = [line.strip() for line in f]
	return documents

# Save an object to a pickle file and provide a message when complete.
def save_obj_to_pkl_file(obj, obj_name, fname):
	with open(fname, 'wb') as f:
		pkl.dump(obj, f, protocol=2)
		logger.info("Saved %s to %s." % (obj_name, fname))


# Save a list to a file, with each element on a newline.
def save_list_to_file(ls, ls_name, fname):
	with codecs.open(fname, 'w', 'utf-8') as f:
		f.write("\n".join(ls))
		logger.debug("Saved %s to %s." % (ls_name, fname))		

# Load an object from a pickle file and provide a message when complete.
def load_obj_from_pkl_file(obj_name, fname):
	with open(fname, 'rb') as f:
		obj = pkl.load(f)
		logger.info("Loaded %s from %s." % (obj_name, fname))		
	return obj

# A TriplesDataseet, comprised of triples.
class TriplesDataset(Dataset):
	def __init__(self, idx, doc_idx, doc, head, relation, tail, y):
		super(TriplesDataset, self).__init__()
		self.idx = idx
		self.doc_idx = doc_idx
		self.doc = doc
		self.head = head
		self.relation = relation
		self.tail = tail
		self.y = y	

	def __getitem__(self, ids):
		return self.idx[ids], self.doc_idx[ids], self.doc[ids], self.head[ids], self.relation[ids], self.tail[ids], self.y[ids]

	def __len__(self):
		return len(self.y)