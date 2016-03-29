# MovieQA

<strong>MovieQA: Understanding Stories in Movies through Question-Answering</strong>  
Makarand Tapaswi, Yukun Zhu, Rainer Stiefelhagen, Antonio Torralba, Raquel Urtasun, and Sanja Fidler  
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, June 2016.  
[Project page](http://movieqa.cs.toronto.edu) |
[arXiv preprint](http://arxiv.org/abs/1512.02902) |
[Read the paper](http://arxiv.org/abs/1512.02902) |
[Explore the data](http://movieqa.cs.toronto.edu/examples/)

----

### Benchmark Data
The data is made available in simple JSON / text files for easy access in any environment. We provide Python scripts to help you get started by providing simple data loaders.

To obtain access to the stories, and evaluate new approaches on the test data, please register at our [benchmark website](http://movieqa.cs.toronto.edu/).


### Python data loader
<code>import MovieQA</code>  
<code>mqa = MovieQA.DataLoader()</code>  

#### Explore
Movies are indexed using their corresponding IMDb keys. For example  
<code>mqa.pprint_movie(mqa.movies_map['tt0133093'])</code>

QAs are stored as a standard Python list
<code>mqa.pprint_qa(mqa.qa_list[0])</code>

#### Use
To get train or test splits of the QA along with a particular story, use  
<code>story, qa = mqa.get_story_qa_data('train', 'plot')</code>

Currently supported story forms are: <code>plot, split plot</code>

----

### Requirements
- numpy
- pysrt


