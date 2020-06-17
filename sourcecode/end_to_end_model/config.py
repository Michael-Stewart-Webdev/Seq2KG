import json, os, codecs


''' logger '''
import codecs
import sys, torch, os

from datetime import datetime
from colorama import Fore, Back, Style
# logger.basicConfig(format=Fore.CYAN + '%(levelname)s: ' + Style.RESET_ALL + '%(message)s', level=logger.DEBUG)
# logger.basicConfig(format=Fore.GREEN + '%(levelname)s: ' + Style.RESET_ALL + '%(message)s', level=logger.INFO)

import data_utils as dutils
from logger import logger, LogToFile
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(125)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(125)



# Create a folder if it does not exist, and return the folder.
def initialise_folder(folder, folder_name):		
	if not os.path.exists(folder):
		os.makedirs(folder)
		logger.info("Created new %s directory %s." % (folder_name, folder))
	return folder

# Check whether a filename exists. Throw an error if it does not. Return the filename if it does.
def check_filename_exists(filename):
	if not os.path.isfile(filename):
		logger.error("%s does not exist." % filename)
		raise IOError()
	return filename

# Dump the config object to a file.

def options_to_text(options):
	return "[" + ", ".join(["%s = %s" % (option, value) for option, value in options.items()]) + "]"


# A config object designed to simplify the process of referring to filenames/folders across multiple locations.
class Config():
	def __init__(self):
		pass

	def load_config(self, dataset):
		with open('config.json', 'r') as f:
			cf = json.load(f)

		self.DATASET  = dataset
		
		self.LEARNING_RATE = cf['learning_rate']
		self.BATCH_SIZE = cf['batch_size']
		self.MAX_SENT_LEN = cf['max_sent_len']
		self.STOP_CONDITION = cf['stop_condition'] # Stop after this many epochs with no f1 improvement

		self.GROUND_TRUTH_TRIPLES_FILE = check_filename_exists("data/%s/ground_truth_triples.csv" % self.DATASET)

		self.TRAIN_FILENAME = "data/%s/train.json" % self.DATASET
		self.DEV_FILENAME = "data/%s/dev.json" % self.DATASET
		self.TEST_FILENAME  = "data/%s/test.txt" % self.DATASET

		self.ANNOTATIONS_FILENAME = "data/%s/annotations.json" % self.DATASET

		self.MODEL_FOLDER 			= initialise_folder("models/%s" % (self.DATASET), "model")
		self.DEBUG_FOLDER 			= initialise_folder("%s/debug" % (self.MODEL_FOLDER), "asset")
		self.ASSET_FOLDER 			= initialise_folder("%s/asset" % (self.MODEL_FOLDER), "asset")
		self.BEST_MODEL_FOLDER 			= initialise_folder("%s/best_model" % (self.MODEL_FOLDER), "best model")
		self.BEST_MODEL_FILENAME		= "%s/model" % self.BEST_MODEL_FOLDER
		self.BEST_MODEL_JSON_FILENAME	= "%s/model.json" % self.BEST_MODEL_FOLDER

		self.EMBEDDING_DIM = cf['embedding_dim']
		self.POSITIONAL_EMB_DIM = cf['positional_emb_dim']
		self.HIDDEN_DIM = cf['hidden_dim']

		# Add the FileHandler to the logger if it doesn't already exist.
		# The logger will log everything to models/<model name>/log.txt.
		if len(logger.root.handlers) == 1:
			t =  datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
			d = initialise_folder("%s/log_%s" % (self.DEBUG_FOLDER, t), "log")
		
			for folder in [d, self.DEBUG_FOLDER]:
				log_filename = "%s/log.txt" % folder
				hdlr = logger.FileHandler(log_filename, 'w+')
				hdlr.setFormatter(LogToFile())
				logger.root.addHandler(hdlr)
				config_dump_filename = "%s/config.txt" % folder
				self.dump_config_to_file(t, config_dump_filename)

	# Dump all of this config object's field variables to the given file.
	# 't' is the current time, which is appended to the top of the file.
	def dump_config_to_file(self, t, fname):
		obj = self.__dict__
		with open(fname, 'w') as f:
			f.write("Config at %s\n" % t)
			f.write("=" * 80)
			f.write("\n")
			for items in sorted(obj.items()):
				f.write(": ".join([str(x) for x in items]))
				f.write("\n")
		logger.debug("Dumped config to %s." % fname)


