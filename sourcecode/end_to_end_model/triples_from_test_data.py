from evaluate import TrainedEndToEndModel

from config import Config
import csv


industries = ['cateringServices', 'automotiveEngineering', 'bbn']

def main():

	for industry in industries:
		print("Evaluating model on %s dataset..." % industry)
		test_data_filename = '../../datasets/%s/test.csv' % industry

		data = []

		# Read the contest data into separate objects
		with open(test_data_filename, 'r') as f:
			reader = csv.reader(f)
			csv_rows = list(reader)
		for row in csv_rows: # skip header
			idx = int(row[0])
			content = row[1]
			data.append(content)

	

		with open('output/%s.csv' % industry, 'w') as f:
			writer = csv.writer(f)
			
			cf = Config()
			cf.load_config(industry)
			E2EModel = TrainedEndToEndModel(industry, cf)
			triples = E2EModel.triples_from_docs(data)

			writer.writerow(['index','s1','r','s2'])
			for doc_idx, doc in enumerate(triples):
				for t in doc:					
					writer.writerow([doc_idx] + [" ".join(t[0])] + [" ".join(t[1])] + [" ".join(t[2])])

if __name__=="__main__":
	main()