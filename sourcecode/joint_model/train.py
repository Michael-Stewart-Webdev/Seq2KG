

import data_utils as dutils
from data_utils import load_documents
from data_utils import Vocab, CategoryHierarchy, EntityTypingDataset, batch_to_wordpieces, wordpieces_to_bert_embs
from logger import logger
from model import E2EETModel
import torch.optim as optim

from progress_bar import ProgressBar
import time, json, sys
import torch
from evaluate import ModelEvaluator
import numpy as np
from sinusoidal_positional_embeddings import *
import pandas as pd

from bert_serving.client import BertClient

from config import Config, device
cf = Config()

torch.manual_seed(125)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(125)
torch.cuda.manual_seed(125)



# Train the model, evaluating it every 10 epochs.
def train(model, data_loaders, word_vocab, wordpiece_vocab, hierarchy_tr, hierarchy_et, ground_truth_triples, epoch_start = 1):

	logger.info("Training model.")
	
	# Set up a new Bert Client, for encoding the wordpieces
	bc = BertClient()


	modelEvaluator = ModelEvaluator(model, data_loaders['dev'], word_vocab, wordpiece_vocab, hierarchy_tr, hierarchy_et, ground_truth_triples, cf)
	
	#optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cf.LEARNING_RATE, momentum=0.9)
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cf.LEARNING_RATE)#, momentum=0.9)
	model.cuda()
	print(cf.LEARNING_RATE)

	num_batches = len(data_loaders["train"])
	max_epochs = 1000
	progress_bar = ProgressBar(num_batches = num_batches, max_epochs = max_epochs, logger = logger)
	avg_loss_list = []

	# Train the model

	for epoch in range(epoch_start, max_epochs + 1):
		epoch_start_time = time.time()
		epoch_losses = []	

		for (i, (batch_x, batch_y_tr, batch_y_et, batch_z_tr, batch_z_et, _, batch_tx, _, _, _)) in enumerate(data_loaders["train"]):
		
			if len(batch_x) < cf.BATCH_SIZE:
				continue
		
			# 1. Convert wordpiece ids into wordpiece tokens
			wordpieces = batch_to_wordpieces(batch_x, wordpiece_vocab)
			wordpiece_embs  = wordpieces_to_bert_embs(wordpieces, bc)

			# 2. Create sin embeddings and concatenate them to the bert embeddings


			wordpiece_embs = wordpiece_embs.to(device)
			batch_y_tr = batch_y_tr.float().to(device)
			batch_y_et = batch_y_et.float().to(device)
			batch_z = batch_z_tr.float().to(device)

			# 3. Feed these vectors to our model
			
			if cf.POSITIONAL_EMB_DIM > 0:
				sin_embs = SinusoidalPositionalEmbedding(embedding_dim=cf.POSITIONAL_EMB_DIM, padding_idx = 0, left_pad = True)
				sin_embs = sin_embs( torch.ones([batch_x.size()[0], batch_x.size()[1]])).to(device)
				joined_embs = torch.cat((wordpiece_embs, sin_embs), dim=2)
			else:
				joined_embs = wordpiece_embs

			# if len(batch_x) < cf.BATCH_SIZE:
			# 	zeros = torch.zeros((cf.BATCH_SIZE - len(batch_x), joined_embs.size()[1], joined_embs.size()[2])).to(device)
			# 	joined_embs = torch.cat((joined_embs, zeros), dim=0)
			# 	print(joined_embs)
			# 	print(joined_embs.size())

			model.zero_grad()
			model.train()

			y_hat_tr, y_hat_et = model(joined_embs)

			loss = model.calculate_loss(y_hat_tr, y_hat_et, batch_x, batch_y_tr, batch_y_et, batch_z)

			# 4. Backpropagate
			loss.backward()
			optimizer.step()
			epoch_losses.append(loss)

			# 5. Draw the progress bar
			progress_bar.draw_bar(i, epoch, epoch_start_time)

		avg_loss = sum(epoch_losses) / float(len(epoch_losses))
		avg_loss_list.append(avg_loss)

		progress_bar.draw_completed_epoch(avg_loss, avg_loss_list, epoch, epoch_start_time)

		modelEvaluator.evaluate_every_n_epochs(1, epoch)




# Convert the ground truth triples csv into a list of lists.
def parse_ground_truth_triples(df):

	ground_truth_triples = []

	current_sent_index = 0
	current_triples = []

	num_dev_documents = len(load_documents(cf.DEV_FILENAME))

	ground_truth_triples_dict = { k: [] for k in range(num_dev_documents) }


	for i, row in enumerate(df.itertuples()):
		sent_index = int(getattr(row, 'index'))
		head = getattr(row, 's1').split()
		rel  = str(getattr(row, 'r')).split()
		tail = getattr(row, 's2').split()
		if sent_index not in ground_truth_triples_dict:
			ground_truth_triples_dict[sent_index] = []
		ground_truth_triples_dict[sent_index].append([head, rel, tail])

	for k in range(num_dev_documents):
		ground_truth_triples.append([])
		for t in ground_truth_triples_dict[k]:
			ground_truth_triples[-1].append(t)
	
	return ground_truth_triples

def main(opts):


	if len(opts) == 0:
		raise ValueError("Usage: train.py <dataset>")
	dataset = opts[0]
	if dataset not in ['cateringServices', 'automotiveEngineering', 'bbn']:
		raise ValueError("Dataset must be either cateringServices, automotiveEngineering, or bbn.")

	cf.load_config(dataset)

	logger.info("Loading files...")


	data_loaders = dutils.load_obj_from_pkl_file('data loaders', cf.ASSET_FOLDER + '/data_loaders.pkl')
	word_vocab = dutils.load_obj_from_pkl_file('word vocab', cf.ASSET_FOLDER + '/word_vocab.pkl')
	wordpiece_vocab = dutils.load_obj_from_pkl_file('wordpiece vocab', cf.ASSET_FOLDER + '/wordpiece_vocab.pkl')
	hierarchy_tr = dutils.load_obj_from_pkl_file('hierarchy_tr', cf.ASSET_FOLDER + '/hierarchy_tr.pkl')
	hierarchy_et = dutils.load_obj_from_pkl_file('hierarchy_et', cf.ASSET_FOLDER + '/hierarchy_et.pkl')
	total_wordpieces = dutils.load_obj_from_pkl_file('total wordpieces', cf.ASSET_FOLDER + '/total_wordpieces.pkl')
	
	ground_truth_triples_df = pd.read_csv(cf.GROUND_TRUTH_TRIPLES_FILE)

	ground_truth_triples = parse_ground_truth_triples(ground_truth_triples_df)
	

	logger.info("Building model.")
	model = E2EETModel(	embedding_dim = cf.EMBEDDING_DIM + cf.POSITIONAL_EMB_DIM,
						hidden_dim = cf.HIDDEN_DIM,
						vocab_size = len(wordpiece_vocab),
						label_size_tr = len(hierarchy_tr),
						label_size_et = len(hierarchy_et),
						total_wordpieces = total_wordpieces,
						max_seq_len = cf.MAX_SENT_LEN,
						batch_size = cf.BATCH_SIZE)
	model.cuda()

	train(model, data_loaders, word_vocab, wordpiece_vocab, hierarchy_tr, hierarchy_et, ground_truth_triples)

if __name__ == "__main__":
	print('hello')
	main(sys.argv[1:])
