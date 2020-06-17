import torch
import torch.nn as nn
from config import device

torch.manual_seed(125)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(125)


class E2EETModel(nn.Module):

	def init_hidden(self):
		# Before we've done anything, we dont have any hidden state.
		# Refer to the Pytorch documentation to see exactly
		# why they have this dimensionality.
		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
		#return (torch.zeros(4, self.batch_size, self.hidden_dim, device=device),
		#		torch.zeros(4, self.batch_size, self.hidden_dim, device=device))

		return (torch.zeros(4, self.batch_size, self.hidden_dim, device=device)) #GRU version

	def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, total_wordpieces, category_counts, hierarchy_matrix, max_seq_len, batch_size):
		super(E2EETModel, self).__init__()

		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size
		self.label_size = label_size

		#self.layer_1 = nn.Linear(embedding_dim, hidden_dim)
		#self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
		
		#self.dropout = nn.Dropout()
		self.hidden2tag = nn.Linear(hidden_dim * 2, label_size)
		self.hidden2tag2 = nn.Linear(label_size, label_size)
		#self.dropout = nn.Dropout(p=0.1)
		self.hierarchy_matrix = hierarchy_matrix
		self.use_hierarchy = False

		self.recurrent_layer = nn.GRU(self.embedding_dim, self.hidden_dim, bidirectional = True, num_layers = 2, dropout = 0.5)

		self.max_seq_len = max_seq_len
		self.batch_size = batch_size

	def forward(self, batch_x):

		self.hidden = self.init_hidden()

		#batch_x = torch.relu(self.layer_1(batch_x))
		#batch_x = self.dropout(torch.relu(self.layer_1(batch_x)))
		#y_hat = self.hidden2tag(batch_x)


		#seq_lens = []
		#for x in batch_x:
		#	seq_lens.append(batch_x.size()[1])
		#print(batch_x.size())
		#batch_x = torch.cat((batch_x, torch.zeros((self.batch_size - batch_x.size()[0], self.max_seq_len, self.embedding_dim)).to(device)))
		#seq_lens = [100, 100, 0, 0, 0, 0, 0, 0, 0, 0]
		
		# GRU
		batch = torch.nn.utils.rnn.pack_padded_sequence(batch_x, [self.max_seq_len] * self.batch_size, batch_first=True, enforce_sorted=False)
		batch, self.hidden = self.recurrent_layer(batch, self.hidden)
		batch, _ = torch.nn.utils.rnn.pad_packed_sequence(batch, batch_first = True)
		batch = batch.contiguous()
		batch = batch.view(-1, batch.shape[2])

		# Feed forward
		#batch = torch.relu(self.layer_1(batch_x))
		#batch = self.dropout(torch.relu(self.layer_1(batch_x)))

		#print(batch.size())

		y_hat = torch.relu(self.hidden2tag(batch))
		y_hat = self.hidden2tag2(y_hat)

	
		
		y_hat = y_hat.view(self.batch_size, self.max_seq_len, self.label_size)


		if self.use_hierarchy:
		  	y_hat_size = y_hat.size()
		  	y_hat_v = y_hat.view(-1, self.hierarchy_matrix.size()[0])
		  	y_hat =  torch.matmul(y_hat_v, self.hierarchy_matrix)
		  	y_hat = y_hat.view(y_hat_size)


		return y_hat
		


	def calculate_loss(self, y_hat, batch_x, batch_y, batch_z):
		non_padding_indexes = torch.ByteTensor((batch_x > 0))
		loss_fn = nn.BCEWithLogitsLoss()
		loss = loss_fn(y_hat[non_padding_indexes], batch_y[non_padding_indexes])
		return loss


	# Predict the labels of a batch of wordpieces using a threshold of 0.5.
	def predict_labels(self, preds):
		#preds_s = torch.sigmoid(preds)		
		hits  = (preds > 0).float()
		return hits

	# Evaluate a given batch_x, predicting the labels.
	def evaluate(self, batch_x):
		preds = self.forward(batch_x)
		return self.predict_labels(preds)


	# Evaluate a given batch_x, but convert the predictions for each wordpiece into the predictions of each token using
	# the token_idxs_to_wp_idxs map.
	def predict_token_labels(self, batch_x, token_idxs_to_wp_idxs):
		preds = self.forward(batch_x)

		avg_preds = torch.zeros(list(batch_x.shape)[0], list(batch_x.shape)[1], list(preds.shape)[2])
	
		for i, batch in enumerate(batch_x):
			for j, wp_idxs in enumerate(token_idxs_to_wp_idxs[i]):		
				avg_preds[i][j] = preds[i][wp_idxs].mean(dim=0)

		return self.predict_labels(avg_preds)

# 5.05 vs 4.81 (e2e, filtering)
