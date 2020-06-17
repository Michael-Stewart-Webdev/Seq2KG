import data_utils as dutils
from logger import logger
from model import CandidateFilteringModel
import torch.optim as optim
from progress_bar import ProgressBar
import time, json
import torch
from config import Config, device
cf = Config()
from evaluate import ModelEvaluator
import pandas as pd
from collections import OrderedDict
from data_utils import load_documents
import sys

# Train the model, evaluating it every 10 epochs.
def train(model, data_loader_train, data_loader_dev, dataset_dev, ground_truth_triples, epoch_start = 1):

	logger.info("Training model.")
	
	modelEvaluator = ModelEvaluator(model, data_loader_dev, dataset_dev, ground_truth_triples, cf)
	
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cf.LEARNING_RATE)#, momentum=0.9)
	model.cuda()

	

	num_batches = len(data_loader_train)
	progress_bar = ProgressBar(num_batches = num_batches, max_epochs = cf.MAX_EPOCHS, logger = logger)
	avg_loss_list = []

	# Train the model
	for epoch in range(epoch_start, cf.MAX_EPOCHS + 1):
		epoch_start_time = time.time()
		epoch_losses = []	

		for (i, (batch_idx, batch_doc_idx, batch_d, batch_h, batch_r, batch_t, batch_y)) in enumerate(data_loader_train):

			# 1. Place each component onto CUDA
			batch_d = batch_d.to(device)
			batch_h = batch_h.to(device)
			batch_r = batch_r.to(device)
			batch_t = batch_t.to(device)
			batch_y = batch_y.float().to(device)

			# 2. Feed these Bert vectors to our model
			model.zero_grad()
			model.train()

			y_hat = model(batch_d, batch_h, batch_r, batch_t)

			# 3. Calculate the loss via BCE
			loss = model.calculate_loss(y_hat, batch_y)

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

	num_dev_documents = len(load_documents(cf.DEV_DOCUMENTS_FILENAME))

	ground_truth_triples_dict = { k: [] for k in range(num_dev_documents) }

	for i, row in enumerate(df.itertuples()):
		sent_index = int(getattr(row, 'index'))
		head = str(getattr(row, 's1')).split()
		rel  = str(getattr(row, 'r')).split()
		tail = str(getattr(row, 's2')).split()
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
		raise ValueError("Usage: evaluate.py <dataset>")
	dataset = opts[0]	
	if dataset not in ['cateringServices', 'automotiveEngineering', 'bbn']:
		raise ValueError("Dataset must be either cateringServices, automotiveEngineering or bbn.")

	cf.load_config(dataset)

	logger.info("Loading data loaders...")

	data_loader_train = dutils.load_obj_from_pkl_file('data loader (train)', cf.ASSET_FOLDER + '/data_loader_train.pkl')
	data_loader_dev   = dutils.load_obj_from_pkl_file('data loader (dev)', cf.ASSET_FOLDER + '/data_loader_dev.pkl')
		
	dataset_dev  = pd.read_csv(cf.DEV_FILENAME)
	ground_truth_triples_df = pd.read_csv(cf.GROUND_TRUTH_TRIPLES_FILE)

	ground_truth_triples = parse_ground_truth_triples(ground_truth_triples_df)

	logger.info("Building model.")
	model = CandidateFilteringModel(
						embedding_dim = cf.EMBEDDING_DIM,
						hidden_dim = cf.HIDDEN_DIM,
						)
	model.cuda()

	train(model, data_loader_train, data_loader_dev, dataset_dev, ground_truth_triples)

if __name__ == "__main__":
	main(sys.argv[1:])
