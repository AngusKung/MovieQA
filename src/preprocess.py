import sys
import os
import argparse
import math
import pdb

import numpy as np
import cPickle as pickle

sys.path.append('/home/MLDickS/MovieQA_benchmark/.')

"""my .py file"""
#from parse import mk_newgru300
#from mybiGRU import findMaxlen


"""Provided by MovieQA"""
from data_loader import DataLoader
from story_loader import StoryLoader

parser = argparse.ArgumentParser()
#parser.add_argument("-path",type=str,required=True)
parser.add_argument("-split",type=str, choices=['train','val','test','trainval','full'],required=True)
parser.add_argument("-type", type=str, choices=['plot','dvs','subtitle','script','split_plot'] , required=True)
args = parser.parse_args()


		
print "Calling word preprocessing on MoiveQA"+" split "+args.split+" type "+args.type

print "Loading MovieQA data..."
DL = DataLoader()
story, qa = DL.get_story_qa_data(args.split,args.type)
movie_list = DL.get_split_movies(args.split)

outpath = '../wholepass.'+args.split+'.'+args.type
print "Writing to "+outpath
fo = open(outpath,'wb')
count = 0 
for imdb_key in movie_list:
	print imdb_key

	whole_pass = ' '.join(story[imdb_key])
	fo.write(whole_pass+'\n')
	count += 1
