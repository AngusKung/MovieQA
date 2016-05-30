import pdb
import math
import numpy as np
import cPickle as pickle
from gensim.models import Word2Vec
from itertools import permutations
import sys

sys.path.append('/home/MLDickS/MovieQA_benchmark/.')
from parse import mk_newgru300
from data_loader import DataLoader
from mybiGRU import findMaxlen
from keras.preprocessing.sequence import pad_sequences

pickBestNum = 5
cosinSim = 0.75

split = 'val' #'train' OR 'val' OR 'test' OR 'full'
story_type='plot' #'plot', 'subtitle', 'dvs', 'script'

wordvec_file = '../GloVe/glove.6B.300d.txt'

DL = DataLoader()
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
    if tempList:
	return tempList
    else:
        return [np.zeros(300,dtype='float32')]

print "Loading word2vec..."
word_vec = Word2Vec.load_word2vec_format(wordvec_file, binary=False)

passages = []
questions = []
A1 = []
A2 = []
A3 = []
A4 = []
A5 = []

numDict = dict()
data = []
answ = []
idxCounter = 0
lastNum = 0
for aQ in qa:
    #pdb.set_trace()
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
    '''oneQ.append(np.vstack(sen_chosenWords))
    oneQ.append(np.vstack(que_wordList))
    oneQ.append(np.vstack(ans1))
    oneQ.append(np.vstack(ans2))
    oneQ.append(np.vstack(ans3))
    oneQ.append(np.vstack(ans4))
    oneQ.append(np.vstack(ans5))'''
    passages.append(sen_chosenWords)
    questions.append(que_wordList)
    A1.append(ans1)
    A2.append(ans2)
    A3.append(ans3)
    A4.append(ans4)
    A5.append(ans5)
    ansC = np.zeros((5,),dtype='float32')
    ansC[aQ[3]] += 1.
    answ.append(ansC)
    #data.append(np.vstack(oneQ))
    '''numDict[str(idxCounter)] = [lastNum,lastNum+len(sen_chosenWords),lastNum+len(sen_chosenWords)+len(que_wordList),
		    lastNum+len(sen_chosenWords)+len(que_wordList)+len(ans1),
		    lastNum+len(sen_chosenWords)+len(que_wordList)+len(ans1)+len(ans2),
		    lastNum+len(sen_chosenWords)+len(que_wordList)+len(ans1)+len(ans2)+len(ans3),
		    lastNum+len(sen_chosenWords)+len(que_wordList)+len(ans1)+len(ans2)+len(ans3)+len(ans4),
		    lastNum+len(sen_chosenWords)+len(que_wordList)+len(ans1)+len(ans2)+len(ans3)+len(ans4)+len(ans5)]
    lastNum += len(sen_chosenWords)+len(que_wordList)+len(ans1)+len(ans2)+len(ans3)+len(ans4)+len(ans5)'''
    idxCounter += 1


'''print "Pickling to ...",dataPickle_name
#pdb.set_trace()
print "pickling..."
fh =open(dataPickle_name,'wb')
pickle.dump(numDict,fh,pickle.HIGHEST_PROTOCOL)
fh.close()'''


maxlen = findMaxlen(A1)
maxlen = findMaxlen(A2,maxlen)
maxlen = findMaxlen(A3,maxlen)
maxlen = findMaxlen(A4,maxlen)
maxlen = findMaxlen(A5,maxlen)
maxlen = findMaxlen(questions,maxlen)
print "MAX_len A&Q  : "+str(maxlen)
maxlen_pass = findMaxlen(passages)
print "MAX_len pass : "+str(maxlen_pass)
Qnum = len(passages)
print "Qnum : "+str(Qnum)

print "Start padding..."
passages = pad_sequences(passages, maxlen=maxlen_pass, dtype='float32')
questions = pad_sequences(questions, maxlen=maxlen, dtype='float32')
A1 = pad_sequences(A1, maxlen=maxlen, dtype='float32')
A2 = pad_sequences(A2, maxlen=maxlen, dtype='float32')
A3 = pad_sequences(A3, maxlen=maxlen, dtype='float32')
A4 = pad_sequences(A4, maxlen=maxlen, dtype='float32')
A5 = pad_sequences(A5, maxlen=maxlen, dtype='float32')
print "Start vstacking..."
mem_ans = np.vstack(answ)

path_name = "../Memmap/"+str(split)+"."+str(story_type)+"."+str(pickBestNum)+".mp="+str(maxlen_pass)+".m="+str(maxlen)+".Q="+str(len(A5))+".lstm/"
passMemmap_name = path_name+"pass.memmap"
queMemmap_name = path_name+"que.memmap"
A1Memmap_name = path_name+"A1.memmap"
A2Memmap_name = path_name+"A2.memmap"
A3Memmap_name = path_name+"A3.memmap"
A4Memmap_name = path_name+"A4.memmap"
A5Memmap_name = path_name+"A5.memmap"
ansMemmap_name = path_name+"ans.memmap"
print "Memmap to ...",path_name
#pdb.set_trace()
print "memmapping..."
pass_fp = np.memmap(passMemmap_name, dtype='float32', mode='w+', shape=(Qnum,maxlen_pass,300))
pass_fp[:,:,:] = np.swapaxes(np.swapaxes(np.dstack(passages),1,2),0,1)[:,:,:]

que_fp = np.memmap(queMemmap_name, dtype='float32', mode='w+', shape=(Qnum,maxlen,300))
que_fp[:,:,:] = np.swapaxes(np.swapaxes(np.dstack(questions),1,2),0,1)[:,:,:]

A1_fp = np.memmap(A1Memmap_name, dtype='float32', mode='w+', shape=(Qnum,maxlen,300))
A1_fp[:,:,:] = np.swapaxes(np.swapaxes(np.dstack(A1),1,2),0,1)[:,:,:]

A2_fp = np.memmap(A2Memmap_name, dtype='float32', mode='w+', shape=(Qnum,maxlen,300))
A2_fp[:,:,:] = np.swapaxes(np.swapaxes(np.dstack(A2),1,2),0,1)[:,:,:]

A3_fp = np.memmap(A3Memmap_name, dtype='float32', mode='w+', shape=(Qnum,maxlen,300))
A3_fp[:,:,:] = np.swapaxes(np.swapaxes(np.dstack(A3),1,2),0,1)[:,:,:]

A4_fp = np.memmap(A4Memmap_name, dtype='float32', mode='w+', shape=(Qnum,maxlen,300))
A4_fp[:,:,:] = np.swapaxes(np.swapaxes(np.dstack(A4),1,2),0,1)[:,:,:]

A5_fp = np.memmap(A5Memmap_name, dtype='float32', mode='w+', shape=(Qnum,maxlen,300))
A5_fp[:,:,:] = np.swapaxes(np.swapaxes(np.dstack(A5),1,2),0,1)[:,:,:]

fp2 = np.memmap(ansMemmap_name, dtype='float32', mode='w+', shape=(len(passages),5))
fp2[:,:] = mem_ans[:,:]
pdb.set_trace()

'''for i in range(0,len(passages)):
    pass_fp[i] = passages[i]
    mem_data = np.vstack([np.vstack(passages[i]),np.vstack(questions[i]),np.vstack(A1[i]),np.vstack(A2[i]),np.vstack(A3[i]),np.vstack(A4[i]),np.vstack(A5[i]) ])
    fp[i*singleQ:(i+1)*singleQ,:] = mem_data
    if i%50==0:
	print i
        pdb.set_trace()
del passages
del questions
del A1
del A2
del A3
del A4
del A5
'''
