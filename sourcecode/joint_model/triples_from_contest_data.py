from evaluate import TrainedEndToEndModel

from config import Config
import csv
contest_data_filename = '../../datasets/icdm_contest_data.csv'

industries = ['cateringServices']

def main():

	data = {
		k: [] for k in industries
	}
	skipped_industries = set()

	# Read the contest data into separate objects
	with open(contest_data_filename, 'r') as f:
		reader = csv.reader(f)
		csv_rows = list(reader)
	for row in csv_rows[1:]: # skip header
		idx = int(row[0])
		content = row[1]
		industry = row[2]
		if industry not in data:
			skipped_industries.add(industry)
		else:
			data[industry].append(content)
			assert len(data[industry]) == (idx + 1)
	if len(skipped_industries) > 0:
		print("Note: the following industries were skipped: %s" % str(skipped_industries))

	with open('output/submission.csv', 'w') as f:
		writer = csv.writer(f)
		for industry in industries:		
			cf = Config()
			cf.load_config(industry)
			E2EModel = TrainedEndToEndModel(industry, cf)
			triples = E2EModel.triples_from_docs(data[industry])

			writer.writerow(['industry','index','s1','r','s2'])
			for doc_idx, doc in enumerate(triples):
				for t in doc:					
					writer.writerow([industry] + [doc_idx] + [" ".join(t[0])] + [" ".join(t[1])] + [" ".join(t[2])])

if __name__=="__main__":
	main()