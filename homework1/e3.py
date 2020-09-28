
import nltk
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
import matplotlib.pyplot as plt
import matplotlib as mpl
import spacy
nlp = spacy.load("en_core_web_sm")
def lemmatizeAndPrint(raw_list):
    print("{0:20}{1:20}".format("Word","Lemma"))
    lemma_list=[]
    for word in raw_list:
        #lemma=wordnet_lemmatizer.lemmatize(word)
        lemma=word.lemma_
        lemma_list.append(lemma)
        print ("{0:}  {1:}".format(word,lemma))

def countAndPlotTypes(raw_list):
    dic_lemma = {}
    for word in raw_list:
        lemma=word.lemma_
        #lemma=wordnet_lemmatizer.lemmatize(word)
        dic_lemma[lemma] =lemma
    name_list=["origin","lemma"]
    num_list=[len(raw_list),len(dic_lemma)]
    print(num_list)
    plt.bar(range(len(num_list)), num_list, color='rgb', tick_label=name_list,alpha=0.5)
    plt.title('word types comparison') 
    plt.show()

if __name__ == "__main__":
    #raw_list=[]
    with open('sorted_types.txt','r') as f:
        line = f.read().strip()
        raw_list = nlp(line.replace('\n',' '))

    lemmatizeAndPrint(raw_list)
    countAndPlotTypes(raw_list)
    #nltk.corpus.gutenberg.fileids()

