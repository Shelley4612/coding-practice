# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import re 

class Tfidf:
    def __init__(self, list_docs):
        self.list_docs = list_docs
        self.n = len(list_docs)  # n, total num of docs
        self.doc_len, self.tokenized_docs, self.vocabulary = self.tokenize()
             
    def tokenize(self):
        for i in range(self.n):
            self.list_docs[i] = self.list_docs[i].lower()
            self.list_docs[i] = re.sub(r'\W', ' ', self.list_docs[i]) #remove punctuations
            self.list_docs[i] = re.sub(r'\s+',' ', self.list_docs[i]) #remove the whitespace from removed punc       
        tokenized_docs = [doc.split() for doc in self.list_docs] # produce list of lists of words
        doc_len = [len(doc) for doc in tokenized_docs]
        vocabulary = set([term for sublists in tokenized_docs for term in sublists]) ##remove the inner lists, then unique
        return doc_len,tokenized_docs,vocabulary
    
    def tf(self):
        # freq(t)/len
        word_count_matrix = []
        for doc in self.tokenized_docs:
            word_count_list = []
            for word in self.vocabulary:
                word_count = len([token for token in doc if token == word])
                word_count_list.append(word_count) # word count for each doc           
            word_count_matrix.append(word_count_list)
        
        BOW_matrix = np.vstack(word_count_matrix)
        tf_matrix = BOW_matrix/np.array(self.doc_len).reshape(self.n,1)
        return tf_matrix    
    
    def idf(self):
         # log(n/(#docs containing the term + 1))
        
        n_docs_t_list = []
        for word in self.vocabulary:
            n_docs_t = 0
            for doc in self.tokenized_docs:
                if word in doc:
                    n_docs_t +=1
            n_docs_t_list.append(n_docs_t)# store a list of # docs(t) for the same length of the vocabulary
        
        idf_matrix = np.log(self.n/(np.array(n_docs_t_list) + 1)).reshape(1,len(self.vocabulary))
        return idf_matrix
 
    def fit(self):
        return self.tf()*self.idf()
        

#### test Tfidf class
        
corpus = [
     'This is the first document.',
    'This document is the second document.',
     'And this is the third one.',
    'Is this the first document?',]
    

t = Tfidf(corpus)
t.tf()
t.idf()
t.fit()

