import pdb
import math
import numpy as np
import cPickle as pickle
from gensim.models import Word2Vec
from itertools import permutations
import sys

sys.path.append('/home/MLDickS/MovieQA_benchmark/.')
from data_loader import DataLoader
from story_loader import StoryLoader

pickBestNum = 5
cosinSim = 0.75

split = 'val' #'train' OR 'val' OR 'test' OR 'full'
story_type='plot' #'plot', 'subtitle', 'dvs', 'script'

wordvec_file = '../GloVe/glove.6B.300d.txt'
dataPickle_name = "../Pickle/"+str(split)+"."+str(story_type)+".lstm.pickle"
print "Saving to : ",dataPickle_name

DL = DataLoader()
SL = StoryLoader()
story,qa = DL.get_story_qa_data(split,story_type)

def wordToVec(entry):
    tempList = []
    for i in range(1): #null loop, just for the right indent
	temp_vector = np.zeros(300,dtype='float32')
	for word in entry:
	    word = word.lower()
	    if word not in word_vec:
		if '\'s' in word:
		    word = word.split('\'')[0]
		elif 'n\'t' in word:
		    temp_vector = np.asarray(word_vec[word.split('n')[0]])
		    word = 'not'
		elif '\'d' in word:
		    temp_vector = np.asarray(word_vec[word.split('\'')[0]])
		    word = 'would'
		elif 'i\'m' in word:
		    temp_vector = np.asarray(word_vec['i'])
		    word = 'am'
		elif '\'ll' in word:
		    temp_vector = np.asarray(word_vec[word.split('\'')[0]])
		    word = 'will'
		elif '\'ve' in word:
		    temp_vector = np.asarray(word_vec[word.split('\'')[0]])
		    word = 'have'
		elif '\'re' in word:
		    temp_vector = np.asarray(word_vec[word.split('\'')[0]])
		    word = 'are'
		elif '(' in word:
		    word = word.split('(')[1]
		elif ')' in word:
		    word = word.split(')')[0]
		elif '.'  in word:
		    for oneword in word.split('.'):
			if oneword and oneword in word_vec:
			    temp_vector = np.asarray(word_vec[oneword])
			    tempList.append(temp_vector)
		    continue
		elif ';' in word:
		    for oneword in word.split(';'):
			if oneword and oneword in word_vec:
			    temp_vector = np.asarray(word_vec[oneword])
			    tempList.append(temp_vector)
		    continue
		elif ':' in word:
		    for oneword in word.split(':'):
			if oneword and oneword in word_vec:
			    temp_vector = np.asarray(word_vec[oneword])
			    tempList.append(temp_vector)
		    continue
		elif '\'' in word:
		    for oneword in word.split('\''):
			if oneword and oneword in word_vec:
			    temp_vector = np.asarray(word_vec[oneword])
			    tempList.append(temp_vector)
		    continue
		elif '-'  in word:
		    for oneword in word.split('-'):
			if oneword and oneword in word_vec:
			    temp_vector = np.asarray(word_vec[oneword])
			    tempList.append(temp_vector)
		    continue
	    try:
		temp_vector = np.add(temp_vector,np.asarray(word_vec[word]))
		tempList.append(temp_vector)
	    except:
		print word
    return tempList

print "Loading word2vec..."
word_vec = Word2Vec.load_word2vec_format(wordvec_file, binary=False)

data = []
for aQ in qa:
    print aQ[0]
    oneQ = []
    que_wordList = wordToVec(str(aQ[1]))
    que_averageList = np.mean( np.asarray(que_wordList,dtype='float32'), axis=0 )
    ans1 =wordToVec(str(aQ[2][0]))
    ans2 =wordToVec(str(aQ[2][1]))
    ans3 =wordToVec(str(aQ[2][2]))
    ans4 =wordToVec(str(aQ[2][3]))
    ans5 =wordToVec(str(aQ[2][4]))
    passage = story[aQ[4]]
    sen_wordList = []
    sen_cosinSim = []
    for sentence in passage:
	theList = wordToVec(sentence)
	sen_wordList.append(theList)
	average_sen = np.mean( np.asarray(theList,dtype='float32'), axis=0 )
	sen_cosinSim.append( np.dot(average_sen,que_averageList.T)/( math.sqrt(np.dot(average_sen,average_sen))*math.sqrt(np.dot(que_averageList,que_averageList)) )  )
    sen_chosenWords = []
    num = 0
    #while sen_cosinSim:
    while num<pickBestNum  and sen_cosinSim:
	#if max(sen_cosinSim) < cosinSim:
	#    break
	print max(sen_cosinSim)
	idx = sen_cosinSim.index(max(sen_cosinSim))
	for word in sen_wordList[idx]:
	    sen_chosenWords.append(word)
	sen_cosinSim.pop(idx)
	sen_wordList.pop(idx)
	num+=1
    oneQ.append(sen_chosenWords)
    oneQ.append(que_wordList)
    oneQ.append(ans1)
    oneQ.append(ans2)
    oneQ.append(ans3)
    oneQ.append(ans4)
    oneQ.append(ans5)
    ansC = np.zeros((5,),dtype='float32')
    ansC[aQ[3]] += 1.
    data.append([oneQ,ansC])


'''print "Pickling to ...",dataPickle_name
pdb.set_trace()
print "pickling..."
fh =open(dataPickle_name,'wb')
pickle.dump(data,fh,pickle.HIGHEST_PROTOCOL)
fh.close()'''

print "Memmap to ...",dataPickle_name
pdb.set_trace()
print "memmapping..."
fp = np.memmap(dataPickle_name, dtype='float32', mode='w+', shape=(len(data),4))
