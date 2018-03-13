import nltk

from nltk.corpus import brown

import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import numpy as np

from collections import Counter

import operator
import random
import time

def preprocess():
	words=brown.words()
	#Brown word corpus 1m words
	#56,057 Vocab size total 
	#49,815 after lowercase
	
	words=[word.lower() for word in words]
	
	word_counts=Counter(words)
	
	trimmed_words=[word for word in words if word_counts[word]>5]
	return trimmed_words
	
def lookup_tables(trimmed_words):
	trimmed_word_counts=Counter(trimmed_words)
	
	sorted_vocab=sorted(trimmed_word_counts.items(), key=operator.itemgetter(1), reverse=True)	
	
	word2idx={word[0]:idx for idx,word in enumerate(sorted_vocab)}
	idx2word={idx:word for word,idx in word2idx.items()}
	
	
	sorted_vocab_dict={word:count for word,count in sorted_vocab}
	
	print sorted_vocab_dict['husband']
	print sorted_vocab_dict['wife']
	print sorted_vocab_dict['king']
	print sorted_vocab_dict['queen']
	print sorted_vocab_dict['the']
	
	
	return word2idx,idx2word
	
	
def get_target(words,idx,window_size=5):
	R=np.random.randint(1,window_size+1)
		
	#R words from history and R words from future
	
	start=idx-R if (idx-R)>0 else 0
	end=idx+R
	
	target_words=set(words[start:idx] + words[idx+1:end+1])
	
	return list(target_words)
	
	
def get_batches(words, batch_size, window_size=5):
	#Creates batches of (Inputs,Targets)
	
	n_batches=len(words)//batch_size #// is floor division
	
	words=words[:n_batches*batch_size] #Full batches only
	
	for idx in range(0,len(words),batch_size):
		x,y=[],[]
		
		batch=words[idx:idx+batch_size]
		
		for i in range(len(batch)):
			batch_x=batch[i]
			batch_y=get_target(batch,i,window_size)
			
			y.extend(batch_y)
			x.extend([batch_x]*len(batch_y))
		
		yield x,y
		
	
	
		
if __name__=='__main__':
	
	trimmed_words=preprocess()
	print "Trimmed words:",len(trimmed_words)
	word2idx,idx2word=lookup_tables(trimmed_words)
	
	int_words=[word2idx[word] for word in trimmed_words]
	
	idx_word_counts=Counter(int_words)
	
	total_count=float(len(int_words))
	
	print "Total = ",total_count
	
	#Subsampling: remove frequent words like the,and etc.
	
	freq={idx:count/total_count for idx,count in idx_word_counts.items()}
	
	threshold=1e-5
	
	p_drop={idx: 1-np.sqrt(threshold/freq[idx]) for idx in idx_word_counts}
	
	train_words=[idx for idx in int_words if p_drop[idx]<random.random()]
	
	print len(train_words) #146,753 training corpus
	
	#Computation Graph
	
	train_graph=tf.Graph()
	
	with train_graph.as_default():
		inputs=tf.placeholder(tf.int32, [None], name='inputs')
		labels=tf.placeholder(tf.int32, [None,None], name='labels')
		
	n_vocab=len(idx2word) #12,416
	
	n_embedding=100 #300
	
	
	with train_graph.as_default():
		embedding=tf.Variable(tf.random_uniform((n_vocab,n_embedding),-1,1))
		embed=tf.nn.embedding_lookup(embedding,inputs)
		
		
	#Negative Sampling
	
	n_sampled=5 #100
	
	with train_graph.as_default():
		softmax_w=tf.Variable(tf.truncated_normal((n_vocab,n_embedding)))
		
		softmax_b=tf.Variable(tf.zeros(n_vocab), name='softmax_bias')
		
		loss=tf.nn.sampled_softmax_loss(weights=softmax_w, biases=softmax_b, labels=labels, inputs=embed, num_sampled=n_sampled, num_classes=n_vocab)
		
		cost=tf.reduce_mean(loss)
		
		optimizer=tf.train.AdamOptimizer().minimize(cost)
		
		
	#Training
	
	epochs=10
	batch_size=1000
	window_size=5 #10
	
	with train_graph.as_default():
		saver=tf.train.Saver()
		
	with tf.Session(graph=train_graph) as sess:
		iteration =1
		loss=0
		sess.run(tf.initialize_all_variables())
		
		for e in range(1, epochs+1):
			batches=get_batches(train_words, batch_size, window_size)
			start=time.time()
			
			for x,y in batches:
				feed={inputs:x, labels:np.array(y)[:,None]}
				train_loss,_=sess.run([cost,optimizer], feed_dict=feed)
				
				loss+=train_loss
				
				if iteration%100==0:
					end=time.time()
					print "Epochs:",e," Iteration:",iteration, " Avg training loss:",loss/100, " sec/batch:",(end-start)/100
					loss=0
					start=time.time()
	
		
				iteration+=1
				
				
		embed_mat=sess.run(embedding)
		
		
		
	#Visualizing the word vectors
	viz_words=100
	tsne=TSNE()
	embed_tsne=tsne.fit_transform(embed_mat[:viz_words,:])
	
	
	fig,ax=plt.subplots(figsize=(14,14))
	
	for idx in range(viz_words):
		plt.scatter(*embed_tsne[idx,:])
		plt.annotate(idx2word[idx],(embed_tsne[idx,0],embed_tsne[idx,1]),alpha=0.7)	
	
	plt.savefig('word2vec_3.png')
	plt.show()
	

