import os
import csv
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from nltk.parse.corenlp import CoreNLPParser
import nltk
from nltk.corpus import stopwords

# Start CoreNLP server first
parser = CoreNLPParser(url='http://localhost:9000')
nltk.download("stopwords")

pdfdirectory = r'data/'
directory = os.listdir(pdfdirectory)
N = len(directory)

dictionaries = [{} for _ in range(N)]
vocabulary = {}

stop_words = set(stopwords.words("english"))

def add_record(word, d):
	#print(word)
	global dictionaries
	t = dictionaries[d].get(word)
	if t is not None:
		t += 1
		dictionaries[d].update({ word: t })
	else:
		dictionaries[d].update({ word: 1 })
	global vocabulary
	v = vocabulary.get(word)
	if v is not None:
		v += 1
		vocabulary.update({ word: v })
	else:
		vocabulary.update({ word: 1 })

def extract_text(pdf_path, d):
	for page_layout in extract_pages(pdf_path):
		page_text = []
		for element in page_layout:
			if isinstance(element, LTTextContainer):
				page_text.append(element.get_text().strip())
		# Remove header and footer lines
		for chunk in page_text:
			if len(chunk) > 0:
				parse = next(parser.raw_parse(chunk))
			for item in parse.leaves():
                #remove stop words
				if item not in stop_words and 'http' not in item:
					item = item.lower()
					add_record(item, d)
                    
from math import log

def probability(user_words):
    #calculates document probabilities using log probabilities with laplace smoothing
    doc_scores = {}

    for i, doc_name in enumerate(directory):
        total_words = sum(dictionaries[i].values())  
        score = 0  

        for word in user_words:
            word_count = dictionaries[i].get(word, 0)
            #prob with laplace smoothing  
            probability = (word_count + 1) / (total_words + len(vocabulary))
            score += log(probability)  #add probs in log (multiply regular probs together but don't lose digits)

        doc_scores[doc_name] = score  

    return doc_scores

from collections import defaultdict

def ngramprob(user_words, n=2):
    #tokenize user words
    tokens = list(parser.tokenize(" ".join(user_words))) 
    #crerate n grams length n
    user_ngrams = list(zip(*[tokens[i:] for i in range(n)]))
    #dict to store n-gram prob for each doc
    ngram_probs = {}
    #for each doc
    for i, doc_name in enumerate(directory):
        total_words = sum(dictionaries[i].values()) #total number of words in doc
        #dict to store word pair probs
        doc_ngram_probs = {}
        #for every ngram from user input
        for ngram in user_ngrams:
            w0, w1 = ngram #ngram 1 and ngram 2
            #occurances of first word
            w0_count = dictionaries[i].get(w0, 0) 
            #occurances of second word
            w1_given_w0_count = dictionaries[i].get(w1, 0)
            #prob with laplace smoothing
            p = (w1_given_w0_count + 1) / (w0_count + len(vocabulary))
          
            second_word_dictionary = doc_ngram_probs.get(w0, {}) # default to empty dictionary returned if no w0 word exists as a key yet
            second_word_dictionary.update({w1: p})
            doc_ngram_probs.update({w0: second_word_dictionary})

        #sort
        for key, distribution in doc_ngram_probs.items():
            sorted_probs = dict(sorted(distribution.items(), key=lambda w_prob: w_prob[1], reverse=True))
            doc_ngram_probs.update({key: sorted_probs})
        #store probs for each doc
        ngram_probs[doc_name] = doc_ngram_probs

    return ngram_probs


print("\n--- PROCESSING DOCUMENTS ---\n")

# process each pdf
for i, pdf_file in enumerate(directory):
    print(f'Processing file {i + 1}/{N}: {pdf_file}')
    extract_text(os.path.join(pdfdirectory, pdf_file), i)

print("\n--- SAVING RESULTS TO CSV ---\n")

# create csv to store results
output_file = 'freq_counts.csv'
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

    # column titles
    col_titles = ['Vocabulary', 'Total Counts'] + directory
    writer.writerow(col_titles)

    # print word frequncies in sorted
    for word in sorted(vocabulary):
        row = [word, vocabulary[word]] + [dictionaries[i].get(word, 0) for i in range(N)]
        writer.writerow(row)

    #total count ea word in doc
    doc_total = [sum(dic.get(word, 0) for dic in dictionaries) for word in sorted(vocabulary)]

    #total count all words in doc
    total_counts_per_doc = [sum(dic.values()) for dic in dictionaries] #sum accross each doc

    total_row = ['TOTAL']+['']+ total_counts_per_doc
    writer.writerow(total_row)

print("\n--- USER INPUT ---\n")
#get user input
user_input = input("Enter comma seperated words ").strip().lower()
user_words = [word.strip() for word in user_input.split(",") if word.strip()]
#call prob function with user words
scores = probability(user_words)  

print("\n--- DISPLAYING ---\n")
#for every doc print log prob
for doc in directory:  
    print(f"Probability for {doc}: {scores[doc]:.8f}") 

#create list for ez sorting of tuples using scores dict
documents = list(scores.items())  
#sort probs in documents
probabilities = [prob for _, prob in documents]  
probabilities.sort(reverse=True)  
#ensure no duplicate probs then join if there is 
for i in range(len(documents)-1):
    if documents[i][1] == documents[i + 1][1]: 
        documents[i] = (f"{documents[i][0]}, {documents[i + 1][0]}", documents[i][1])  
        documents.pop(i + 1) 

#print
print("\n--- Documents ranked by probability ---\n")
for prob in probabilities:
    for doc, p in documents:  
        if p == prob:
            print(f"{doc}: {prob:.8f}")
            break  

print(f"\nThe most likely document to contain the words is {documents[0][0]}")

ngram_scores = ngramprob(user_words, n=2)

#print ngram
print("\n--- N Gram Probabilities ---\n")
print("\n Descending order the document most likely to contain the words are \n")
for doc, probs in ngram_scores.items():
    print(f"\nDocument: {doc}")
    for w0, distribution in probs.items():
        for w1, p in distribution.items():
            print(f"P( {w1:14} | {w0:14} ) = {p:.5f}")

