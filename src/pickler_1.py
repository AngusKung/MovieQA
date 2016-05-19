import pdb
import sys

sys.path.append('/home/MLDickS/MovieQA_benchmark/.')
from data_loader import DataLoader
from story_loader import StoryLoader

DL = DataLoader()
SL = StoryLoader()

story,qa = DL.get_story_qa_data()
data = []

for oneQ in qa:
    que = str(qa[1])

pdb.set_trace()
