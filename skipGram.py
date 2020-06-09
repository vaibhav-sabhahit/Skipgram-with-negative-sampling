#!/usr/bin/env python3

from __future__ import division
import argparse
import pandas as pd
import spacy as sp
from tqdm import tqdm
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize
import os
import pickle

nlp = sp.load("en_core_web_sm")

__authors__ = ['Vaibhav Sabhahit','Osheen Mohan Pannikote','Srijan Goyal','Sunjidma Shagdarsuren']
__emails__  = ['vaibhav.sabhahit@essec.edu','osheenmohan.pannikote@essec.edu','srijan.goyal@essec.edu','sunjidma.shagdarsuren@essec.edu']

#function to preprocess the text 
def text2sentences(path):
    sentences = []
    string=''
    with open(path,encoding="utf8") as f:
        content=f.read()
        docs_raw = content.splitlines()
        for l in (docs_raw):
            x=nlp(l.lower())
            string_tokens = [token.orth_ for token in x if not token.is_punct]
            sentences.append(string_tokens)
        return sentences

#load pairs of data
def loadPairs(path):
    data = pd.read_csv(path, delimiter='\t')
    pairs = zip(data['word1'],data['word2'],data['similarity'])
    return pairs


class SkipGram:
    def __init__(self, sentences, nEmbed=100, negativeRate=4, winSize = 5, minCount = 3):
        self.vocab_old={}
        #obtain vocab dict
        for line in sentences:
            for word in line:
                if word not in self.vocab_old:
                    self.vocab_old[word]=1
                else:
                    self.vocab_old[word]+= 1
        self.vocab = {k:v for k,v in self.vocab_old.items() if (v > minCount)}  #remove words less than mincount
        self.w2id = dict((i,word) for word,i in enumerate(self.vocab)) #making of word to ID
        self.id2cnt = {self.w2id[word]:count for word,count in self.vocab.items()}
        self.sentences = sentences
        self.negativeRate = negativeRate
        self.winSize = winSize
        self.lr = 0.1
        self.weight_1=np.random.uniform(-1,1,(len(self.vocab.keys()),nEmbed)) #initialize the weights
        self.weight_2=np.random.uniform(-1,1,(len(self.vocab.keys()),nEmbed))
        self.weight_inter=np.random.uniform(-1,1,(len(self.vocab.keys()),nEmbed))


    def sample(self, omit):
        negative_samples = list(np.random.choice(list(self.unigram.keys()),size = self.negativeRate, p=list(self.unigram.values())))
        negative_samples = [i for i in negative_samples if i not in omit]#check if the generated word is not context word or target word
        return negative_samples
     
    def train(self):
        self.createunigram()
        for counter,sentence in (enumerate((sentences))):
            sentence = list(filter(lambda word: word in self.vocab, sentence))#check if all words of sentence are in vocab
            for wpos,word in enumerate(sentence):
                wIdx = self.w2id[word]
                winsize = self.winSize
                start = max(0, wpos - winsize)
                end = min(wpos + winsize + 1, len(sentence))
               
                for context_word in sentence[start:end]: #loop through the window to get the context words.
                    ctxtId = self.w2id[context_word]
                    if ctxtId == wIdx: continue
                    negativeIds = self.sample([wIdx, ctxtId])
                    self.trainWord(wIdx, ctxtId, negativeIds)
					
    def trainWord(self, wid, cid, nids):
                
        pnids = [(cid, 1)] + [(nid,0) for nid in nids] #label 1(positive word) label 0(negative word)
        for i in range(10):
            for pnid, t in pnids:
                dot_prod = np.dot(self.weight_1[wid, :], self.weight_2[pnid, :])
                s = self.sigmoid(dot_prod)
                #updating W2
                self.weight_inter[pnid, :] -= self.lr * (s-t) * self.weight_1[wid, :]
                # updating W1
                self.weight_1[wid, :] -= self.lr * (s-t) * self.weight_2[pnid, :]
                self.weight_2[pnid, :] = self.weight_inter[pnid, :]         

    def createunigram(self):
        norm = sum([occurence**0.75 for occurence in self.id2cnt.values()])
        self.unigram = {k:(v**0.75)/norm for k, v in self.id2cnt.items()} #create unigramtable

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def save(self, path):
            path1 = os.path.join(path, 'mymodel.pkl')
            pickle.dump(self,open(path1,'wb'))

    def similarity(self,word1,word2):
            if(word1 not in self.vocab or word2 not in self.vocab): #map words not present in the vocab to zero
                return 0
            else:
                vec1=self.weight_1[self.w2id[word1]]#find the cosine distanse between the two words
                vec2=self.weight_1[self.w2id[word2]]
                vec_sum=np.dot(vec1,vec2)
                vec_norm=np.linalg.norm(vec1)*np.linalg.norm(vec2)
                cosine_dist=vec_sum/vec_norm
                return round(abs(cosine_dist),3)
    
    @staticmethod
    def load(path):
  
            path1 = os.path.join(path, 'mymodel.pkl')
            sg=pickle.load(open(path1,'rb'))
            return(sg)
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='simlex.csv', required=True)
    parser.add_argument('--model', help='mymodel.model', required=True)
    parser.add_argument('--test', help='simlex.csv', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = SkipGram(sentences)
        sg.train()
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)
        sg = SkipGram.load(opts.model)
        for a,b,_ in pairs:
            # make sure this does not raise any exception, even if a or b are not in sg.vocab
            print(sg.similarity(a,b))