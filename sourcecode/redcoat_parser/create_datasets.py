# parse_redcoat_data.py by Michael Stewart
# Converts a Redcoat annotations file, i.e. 'annotations.json', into two datasets:
# - triples dataset, for use in the candidate_filtering model
# - end-to-end dataset, for use in the end_to_end model
from redcoat_to_sents import redcoat_to_sents
from sents_to_triples import create_triples_from_redcoat_annotations, get_documents_from_redcoat_annotations, get_redcoat_triples
import random, jsonlines, json, os
from shutil import copyfile

datasets = ["cateringServices"]#, "automotiveEngineering"] # ,"automotiveEngineering",

# Split a given dataset into 90% training and 10% dev data.
def train_dev_split(dataset):
	random.seed(127)
	random.shuffle(dataset)
	train, dev = dataset[:int(len(dataset) * 0.9)], dataset[int(len(dataset) * 0.9):]
	return train, dev

# Save the sentences dataset to the corresponding train and dev .json files.
def save_sents_dataset(data, ground_truth_triples, dataset_name, dataset):
	print("Saving %d parsed annotations..." % (len(data)))
	with open('output/%s/end_to_end_model/%s.json' % (dataset_name, dataset), 'w') as f:
		for obj in data:
			f.write(json.dumps(obj))
			f.write("\n")

	print("Saving ground truth triples in %s set..." % dataset)
	with open('output/%s/end_to_end_model/ground_truth_triples_%s.csv' % (dataset_name, dataset), 'w') as f:
		f.write("index,s1,r,s2\n")
		for t in ground_truth_triples:
			f.write('"%d","%s","%s","%s"' % (t[0], t[1], t[2], t[3]))
			f.write("\n")

def save_triples_dataset(triples, documents, ground_truth_triples, dataset_name, dataset):
	print("Saving triples dataset...")
		
	with open('output/%s/filtering_model/documents_%s.txt' % (dataset_name, dataset), 'w') as f:
	 	for doc_idx, doc in enumerate(documents):
	 		f.write(doc)
	 		f.write("\n")

	with open('output/%s/filtering_model/%s.csv' % (dataset_name, dataset), 'w') as f:
		f.write("index,s1,r,s2,label\n")
		for t in triples:
			f.write('"%d","%s","%s","%s","%d"' % (t[0], t[1], t[2], t[3], t[4]))
			f.write("\n")

	print("Saving ground truth triples in %s set..." % dataset)
	with open('output/%s/filtering_model/ground_truth_triples_%s.csv' % (dataset_name, dataset), 'w') as f:
		f.write("index,s1,r,s2\n")
		for t in ground_truth_triples:
			f.write('"%d","%s","%s","%s"' % (t[0], t[1], t[2], t[3]))
			f.write("\n")

def init_folders(dataset_name):
	directories = [
		'output/%s' % dataset_name,
		'output/%s/filtering_model' % dataset_name,
		'output/%s/end_to_end_model' % dataset_name
	]
	for d in directories:
		if not os.path.exists(d):
			os.makedirs(d)

def create_docs_dataset(dataset_name, subset):
	with jsonlines.open('data/%s/%s.json' % (dataset_name, subset)) as reader:
		dataset = [obj for obj in reader]

	documents = []
	unlabeled_docs = 0
	for obj in dataset:
		if len(obj['mentions']) == 0:
			unlabeled_docs += 1
			continue
		doc = {}
		doc['tokens'] = obj['tokens']
		doc['mentions'] = []
		if dataset_name == "bbn":
			doc['mentions'] = {}
			doc['mentions']['entity_types'] = obj['mentions']['entity_types']
			doc['mentions']['triples'] = []
			iterator = obj['mentions']['triples']
		else:
			iterator = obj['mentions']
		for m in iterator:
			labels = set()

			for l in m['labels']:	
				if "/" not in l:
					continue
				l = l.split('/')[1]	
				label_type, idx = l.split("_")
				labels.add(label_type)
				labels.add(label_type + '/' + idx)			
			m['labels'] = list(labels)
			if dataset_name == "bbn":
				doc['mentions']['triples'].append(m)
			else:
				doc['mentions'].append(m)

		documents.append(doc)

	print("Note: %d documents were unlabeled." % unlabeled_docs)
	return documents

def create_sents_dataset(dataset_name, dataset):
	
	# 1. Open the annotations file
	with jsonlines.open('data/%s/%s.json' % (dataset_name, dataset)) as reader:
		data = [obj for obj in reader]

	# 2. Create the redcoat sentences dataset
	sents_dataset = redcoat_to_sents(data)


	return sents_dataset

def create_triples_dataset(dataset, dataset_name, extraction_model):
	return  create_triples_from_redcoat_annotations(dataset, dataset_name, extraction_model), \
			get_documents_from_redcoat_annotations(dataset, dataset_name)

def extract_ground_truth_triples(dataset_name):
	pass

# Copy the output files into the appropriate model directories.
def copy_files_to_model_directories(dataset_name):
	print("Copying dataset files into model directories...")
	for (source, destination) in (
		('output/%s/filtering_model/train.csv' % dataset_name, '../candidate_filtering/data/%s/train.csv' % dataset_name),
		('output/%s/filtering_model/dev.csv' % dataset_name, '../candidate_filtering/data/%s/dev.csv' % dataset_name),
		('output/%s/filtering_model/documents_train.txt' % dataset_name, '../candidate_filtering/data/%s/documents_train.txt' % dataset_name),
		('output/%s/filtering_model/documents_dev.txt' % dataset_name, '../candidate_filtering/data/%s/documents_dev.txt' % dataset_name),
		('output/%s/filtering_model/ground_truth_triples_dev.csv' % dataset_name, '../candidate_filtering/data/%s/ground_truth_triples.csv' % dataset_name),
		('output/%s/end_to_end_model/train.json' % dataset_name, '../end_to_end_model/data/%s/train.json' % dataset_name),
		('output/%s/end_to_end_model/dev.json' % dataset_name, '../end_to_end_model/data/%s/dev.json' % dataset_name),
		('output/%s/end_to_end_model/ground_truth_triples_dev.csv' % dataset_name, '../end_to_end_model/data/%s/ground_truth_triples.csv' % dataset_name),
		('output/%s/end_to_end_model/train.json' % dataset_name, '../joint_model/data/%s/train.json' % dataset_name),
		('output/%s/end_to_end_model/dev.json' % dataset_name, '../joint_model/data/%s/dev.json' % dataset_name),
		('output/%s/end_to_end_model/ground_truth_triples_dev.csv' % dataset_name, '../joint_model/data/%s/ground_truth_triples.csv' % dataset_name),

	):
		copyfile(source, destination)

# Retrieve the list of ground truth triples for a given dev set.
def get_ground_truth_triples(dev_set):
	ground_truth_triples = []
	for doc_idx, doc in enumerate(dev_set):
		ground_truth_triples += [[doc_idx] + gt + [1] for gt in get_redcoat_triples(doc)]
	return ground_truth_triples

def main():
	for dataset_name in datasets:
		print("Building %s dataset..." % dataset_name)
		init_folders(dataset_name)

		# Create filtering model data

		
		print("\nCreating triples data for Filtering Model")
		for s in ["train", "dev", "test"]:#["train", "dev"]:
		
			dataset = create_docs_dataset(dataset_name, s)
			
			# ground_truth_triples = get_ground_truth_triples(dataset)	
			

			# triples_dataset, documents = create_triples_dataset(dataset, dataset_name, 'michael')
			# save_triples_dataset(triples_dataset, documents, ground_truth_triples, dataset_name, s)

		

			# Create end-to-end data
			print("\nCreating sentence data for End-to-End Model")

			#dataset = create_docs_dataset(dataset_name, s)
			#if dataset_name != "bbn": # No need to apply coref/sentence splitting to the bbn dataset as it contains short sents
			#	dataset = create_sents_dataset(dataset_name, s)
			#dataset_train, dataset_dev = train_dev_split(dataset)
			ground_truth_triples = get_ground_truth_triples(dataset)

			save_sents_dataset(dataset, ground_truth_triples, dataset_name, s)


		copy_files_to_model_directories(dataset_name)

if __name__== "__main__":
	main()