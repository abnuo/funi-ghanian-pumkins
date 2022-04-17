import textgen
import random
import glob

with open("corpus.txt","r",encoding="utf-8") as f:
  text = f.read().replace("\n","")
images = [list(open(i,"rb").read()) for i in glob.glob("mnist/training/0/*.raw")]
data = textgen.buildImageDataset(images)
print(data)
rnn = textgen.RNN(ds=data)
rnn.loadpkl("brain.pkl")
#rnn.train(1,9000)
rnn.savepkl("brain.pkl")
t = rnn.generate(90)
print("Hey",t)
print("".join([chr(round(i%65536)) for i in t]))
