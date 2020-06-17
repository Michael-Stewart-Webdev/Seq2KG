# coding=UTF-8
import pandas as pd
from datetime import datetime
import argparse
import numpy as np

def is_colname_valid(df): 
	user_columns = df.columns.values.tolist()
	target_columns = set(['industry', 'index', 's1', 's2', 'r'])
	assert target_columns.issubset(user_columns), 'Invalid column names.'

def is_index_integer(df):
	for i, row in df.iterrows():
		assert  isinstance(row['index'],(int, np.integer)) or row['index'].isdigit(),'Invalid index on row: '+ str(i)

def is_tuple_string(df):
	for i, row in df.iterrows():
		assert  isinstance(row['s1'],str) and isinstance(row['s2'],str),'Invalid triplet on row: '+ str(i)

def is_industry_valid(df):
	user_industry = set(list(df['industry']))
	target_industry = set(['cateringServices','cosmetics','publicSecurity','automotiveEngineering'])
	assert user_industry.issubset(target_industry), 'Invalid value in industry column.'

def main(file_path):
	df = pd.read_csv(file_path, encoding='utf-8')
	is_colname_valid(df)
	is_index_integer(df)
	is_tuple_string(df)
	is_industry_valid(df)
	file_name = 'submission_' + str((datetime.now()- datetime(1970, 1, 1)).seconds) + '.csv'
	submission = df[['industry', 'index', 's1', 's2', 'r']]
	df.to_csv(file_name,index = False)
	print('Please check the file in current folder: ' + file_name)
	print('Success!')

if __name__ == '__main__':
	print('Transforming...')
	parser = argparse.ArgumentParser()
	parser.add_argument('file',  type=str, help='Input file.')
	args = parser.parse_args()
	file_path = args.file
	main(file_path)
