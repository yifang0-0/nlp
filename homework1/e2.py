import nltk 
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer 
from nltk.stem import LancasterStemmer
import matplotlib.pyplot as plt
import matplotlib as mpl
wordnet_lemmatizer = WordNetLemmatizer()
#nltk.download()
porter=PorterStemmer()
lancaster=LancasterStemmer()
raw_list=[]
with open('sorted_types.txt','r') as f:
    line = f.read().strip('\n')
    raw_list = line.split("\n")
    
print(raw_list)
# output=[]
# for i in raw:
#     output.append(PorterStemmer.stem(i))
print("{0:20}{1:20}{2:20}".format("Word","Porter Stemmer","lancaster Stemmer"))
for word in raw_list:
    print("{0:20}{1:20}{2:20}".format(word,porter.stem(word),lancaster.stem(word)))
#nltk.corpus.gutenberg.fileids()

porter_list = []
lancaster_list = []
dic_porter = {}
dic_lancaster = {}
for word in raw_list:
    porter_list.append(porter.stem(word))
    lancaster_list.append(lancaster.stem(word))
    dic_porter[porter.stem(word)] =porter.stem(word)
    dic_lancaster[lancaster.stem(word)] =lancaster.stem(word)

name_list=["origin","porter","lancaster"]
num_list=[len(raw_list),len(dic_porter),len(dic_lancaster)]
print(num_list)
plt.bar(range(len(num_list)), num_list, color='rgb', tick_label=name_list,alpha=0.5)
plt.title('word types comparison') 
plt.show()

