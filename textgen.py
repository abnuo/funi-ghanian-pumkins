import random
import pickle
import itertools
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from pybrain.datasets import SupervisedDataSet, UnsupervisedDataSet, SequenceClassificationDataSet, SequentialDataSet
from pybrain.structure.modules import LSTMLayer, SoftmaxLayer, LinearLayer, SigmoidLayer
from pybrain.supervised import RPropMinusTrainer, BackpropTrainer
from pybrain.tools.validation import testOnSequenceData
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader

def chance(l,v):
  return float("0."+str(l.count(v)%100))

def buildCharDataset(text):
  return list(map(ord,text))

def buildWordDataset(text):
  words = word_tokenize(text)
  wordslow = list(map(lambda x: x.lower(),words))
  return (words,[list(set(wordslow)).index(i) for i in wordslow])

def buildImageDataset(images):
  ds = SupervisedDataSet(len(images[0]),len(images[0]))
  for i in images:
    ds.addSample(i,i[1:]+[i[0]])
  return ds

class RNN:
  def __init__(self,data=None,ds=None,words=None):
    if not data == None:
      self.ds = SupervisedDataSet(1,1)
      #for i,n in zip(data,itertools.cycle(data[1:])):
      #  self.ds.addSample(i,n)
      i = [[i] for i in data]
      t = [[i] for i in data[1:]+[data[0]]]
      print(len(i),len(t),i,t)
      #self.ds.addSample(i,t)
      self.ds.setField("input",i)
      self.ds.setField("target",t)
      self.ts = UnsupervisedDataSet(len(data))
      self.ts.setField("sample",data)
    if not ds == None:
      self.ds = ds
    self.net = buildNetwork(self.ds.indim,5,self.ds.outdim,hiddenclass=LSTMLayer,outclass=LinearLayer,bias=False,recurrent=True)
    self.words = words
  
  def loadnet(self,fn):
    self.net = NetworkReader.readFrom(fn)
  
  def savenet(self,fn):
    NetworkWriter.writeToFile(self.net,fn)
  
  def loadpkl(self,fn):
    with open(fn,"rb") as f:
      self.net = pickle.loads(f.read())
  
  def savepkl(self,fn):
    with open(fn,"wb") as f:
      f.write(pickle.dumps(self.net))
  
  def train(self,cycles=100,epochs_per_cycle=2):
    trainer = BackpropTrainer(self.net, dataset=self.ds)
    for i in range(cycles):
      trainer.trainEpochs(epochs_per_cycle)
      trnresult = trainer.testOnData()
      print("train error:",trnresult,"cycle:",i+1,"epochs:",(i+1)*epochs_per_cycle)
  
  def generate(self,l):
    init = random.choice(self.ds.getField("input"))[0]
    print("Initial state:",init)
    s = init
    v = []
    for i in range(l):
      s = self.net.activate(s)[0]
      v.append(s)
    return v
  
  def pickle(self):
    return pickle.dumps("hh")
