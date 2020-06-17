import torch
import torch.nn as nn
from config import device
torch.manual_seed(125)

# A candidate filtering model.
# Its goal is to determine whether a given triple is 'valid' or not (i.e. perform binary classification).
class CandidateFilteringModel(nn.Module):
	def __init__(self, embedding_dim, hidden_dim ):
		super(CandidateFilteringModel, self).__init__()

		self.layer_1 = nn.Linear(embedding_dim * 3, hidden_dim)
		self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
		self.dropout = nn.Dropout(p=0.5)
		self.hidden2label = nn.Linear(hidden_dim, 1)
	
	# Take an input doc embedding, head embedding, relation embedding and tail embedding and send it through the network.
	def forward(self, batch_d, batch_h, batch_r, batch_t):

		combined_representation = torch.cat((batch_h, batch_r, batch_t), dim=1)
		#combined_representation = torch.cat((batch_h, batch_r, batch_t), dim=1)

		batch_x_out = torch.tanh(self.layer_1(combined_representation))
		#batch_x_out = torch.relu(self.layer_1(combined_representation))
		batch_x_out = self.dropout(torch.tanh(self.layer_2(batch_x_out)))			

		y_hat = self.hidden2label(batch_x_out)

		return y_hat.view(-1)		
		
	# The loss calculation function.
	def calculate_loss(self, y_hat, batch_y):
		loss_fn = nn.BCEWithLogitsLoss()
		loss = loss_fn(y_hat, batch_y)
		return loss

	# Predict the labels given the batch inputs (document embedding, head embedding, relation embedding, tail embedding).
	# A label is 0 when the weights of the final layer are are <= 0, and 1 when the weights are > 0.
	def predict_labels(self, batch_d, batch_h, batch_r, batch_t):
		y_hat = self.forward(batch_d, batch_h, batch_r, batch_t)
		y_hat = torch.sigmoid(y_hat)
		preds = (y_hat > 0.33)		
		return preds, y_hat # Return the confidence values as well
