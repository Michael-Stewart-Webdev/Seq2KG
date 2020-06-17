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

	def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size_tr, label_size_et, total_wordpieces, max_seq_len, batch_size):
		super(E2EETModel, self).__init__()

		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size
		self.label_size_tr = label_size_tr
		self.label_size_et = label_size_et

		#self.layer_1 = nn.Linear(embedding_dim, hidden_dim)
		#self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
		
		#self.dropout = nn.Dropout()
		self.hidden2tag_tr = nn.Linear(hidden_dim * 2, label_size_tr)
		self.hidden2tag_et = nn.Linear(hidden_dim * 2, label_size_et)

		self.hidden2tag2_tr = nn.Linear(label_size_tr, label_size_tr)
		self.hidden2tag2_et = nn.Linear(label_size_et, label_size_et)
		#self.dropout = nn.Dropout(p=0.1)

		self.recurrent_layer_tr = nn.GRU(self.embedding_dim, self.hidden_dim, bidirectional = True, num_layers = 2, dropout = 0.5)

		self.recurrent_layer_et = nn.GRU(self.embedding_dim, self.hidden_dim, bidirectional = True, num_layers = 2, dropout = 0.5)

		self.max_seq_len = max_seq_len
		self.batch_size = batch_size

	def forward(self, batch_x):

		self.hidden_tr = self.init_hidden()
		self.hidden_et = self.init_hidden()

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
		batch_tr = torch.nn.utils.rnn.pack_padded_sequence(batch_x, [self.max_seq_len] * self.batch_size, batch_first=True, enforce_sorted=False)
		batch_tr, self.hidden_tr = self.recurrent_layer_tr(batch_tr, self.hidden_tr)
		batch_tr, _ = torch.nn.utils.rnn.pad_packed_sequence(batch_tr, batch_first = True)
		batch_tr = batch_tr.contiguous()
		batch_tr = batch_tr.view(-1, batch_tr.shape[2])


		batch_et = torch.nn.utils.rnn.pack_padded_sequence(batch_x, [self.max_seq_len] * self.batch_size, batch_first=True, enforce_sorted=False)
		batch_et, self.hidden_et = self.recurrent_layer_tr(batch_et, self.hidden_et)
		batch_et, _ = torch.nn.utils.rnn.pad_packed_sequence(batch_et, batch_first = True)
		batch_et = batch_et.contiguous()
		batch_et = batch_et.view(-1, batch_et.shape[2])


		# Feed forward
		#batch = torch.relu(self.layer_1(batch_x))
		#batch = self.dropout(torch.relu(self.layer_1(batch_x)))

		#print(batch.size())

		y_hat_tr = torch.relu(self.hidden2tag_tr(batch_tr))
		y_hat_tr = self.hidden2tag2_tr(y_hat_tr)		
		y_hat_tr = y_hat_tr.view(self.batch_size, self.max_seq_len, self.label_size_tr)


		y_hat_et = torch.relu(self.hidden2tag_et(batch_et))
		y_hat_et = self.hidden2tag2_et(y_hat_et)		
		y_hat_et = y_hat_et.view(self.batch_size, self.max_seq_len, self.label_size_et)


		return y_hat_tr, y_hat_et
		


	def calculate_loss(self, y_hat_tr, y_hat_et, batch_x, batch_y_tr, batch_y_et, batch_z):
		non_padding_indexes = torch.ByteTensor((batch_x > 0))
		loss_fn = nn.BCEWithLogitsLoss()
		loss_tr = loss_fn(y_hat_tr[non_padding_indexes], batch_y_tr[non_padding_indexes])
		loss_et = loss_fn(y_hat_et[non_padding_indexes], batch_y_et[non_padding_indexes])
		return (loss_tr + loss_et) / 2


	# Predict the labels of a batch of wordpieces using a threshold of 0.5.
	def predict_labels(self, preds):
		#preds_s = torch.sigmoid(preds)		
		hits  = (preds > 0).float()
		return hits

	# Evaluate a given batch_x, predicting the labels.
	def evaluate(self, batch_x):
		preds_tr, preds_et = self.forward(batch_x)
		return (self.predict_labels(preds_tr), self.predict_labels(preds_et))



	# Evaluate a given batch_x, but convert the predictions for each wordpiece into the predictions of each token using
	# the token_idxs_to_wp_idxs map.
	def predict_token_labels(self, batch_x, token_idxs_to_wp_idxs):
		preds_tr, preds_et = self.forward(batch_x)
		labels = []
		for preds in (preds_tr, preds_et):

			avg_preds = torch.zeros(list(batch_x.shape)[0], list(batch_x.shape)[1], list(preds.shape)[2])
		
			for i, batch in enumerate(batch_x):
				for j, wp_idxs in enumerate(token_idxs_to_wp_idxs[i]):		
					avg_preds[i][j] = preds[i][wp_idxs].mean(dim=0)

			labels.append(self.predict_labels(avg_preds))
		return labels

# 5.05 vs 4.81 (e2e, filtering)
