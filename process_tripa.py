import glob
import json
import csv
import nltk

def jsontocsv():
	files = glob.glob("json/*.json")
	data_file = open('data_file.csv', 'w', encoding='utf-8')
	for file in files:
		with open(file) as json_file:
			data = json.load(json_file)
		reviews=data['Reviews']
		sentenceid=1
		csv_writer = csv.writer(data_file)
		header=['sentenceid', 'reviewid', 'content']
		csv_writer.writerow(header)
		for review in reviews:
			reviewid=review["ReviewID"]
			content=review["Content"]
			sentences = nltk.sent_tokenize(content)
			for sentence in sentences:
				csv_writer.writerow([sentenceid, reviewid, sentence])
				sentenceid+=1
	data_file.close()
jsontocsv()
