import numpy as np
from encoder import Encoder
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
#nltk.download('stopwords')
#ltk.download('punkt')
#nltk.download('wordnet')




#myenc = Encoder()


def cos_matrix(generated_matrix_of_sentences, given_sentence):
    #For generating matrix of cosine similarities
    #myenc = Encoder()
    emgiven = myenc.encode(given_sentence)
    res = []
    for i in range(len(generated_matrix_of_sentences)):
        emgenerated = myenc.encode(generated_matrix_of_sentences[i])
        a = myenc.cosine_similarity(emgiven, emgenerated)[0][0]
        res.append(a)
        
        #For getting the average of the matrix
        res_np = np.array(res)
        mean = np.mean(res_np)
        minimum = np.min(res_np)
        
    return mean, minimum


def filter_input1(directory):
    with open(directory) as file:
        text = file.read()
        sentence = text.translate({ord(i):None for i in '.!@#$?]}{[,\n'}) #remove all unnecessary elements
    return sentence


def filter_input2(path, stem=True):
    wordnet_lemmatizer = WordNetLemmatizer()
    f = open("./stop.txt", 'r')
    stop = f.read().split('\n')
    stopWords = set(stopwords.words('english'))
    stopWords |= set(stop)

    #path = "./test-updated/" + str(dir_num) + "/input"
    with open(path) as file:
        text = file.read()
        sentence = text.translate({ord(i):None for i in '.!@#$?]}{[,'}) #remove all unnecessary elements
        
    def stem_filter(sentence):
        return " ".join(wordnet_lemmatizer.lemmatize(word.lower(), pos='v') for word in word_tokenize(sentence) if word.lower() not in stopWords)

    def tagged_filter(sentence):
        tagged_sent = nltk.pos_tag(word_tokenize(sentence))
        selected = ['CD', 'FW', 'JJ', 'NN', 'NNP', 'NNS', 'NNPS', 'VBG']
        return " ".join([word[0] for word in tagged_sent if word[1] in selected])
    
    return stem_filter(sentence) if stem else tagged_filter(sentence)



"""
directory = "/home/yerlan/HackNU/images/1/input"
yer = {1: ["Aman", "haah"], 2: ["Yerla", "Salam"]}
given = filter_input1(directory)
zhanik = filter_input2(directory)



#Printing Results without Zhanibek
print("Without Zhanibek")

for i in range(len(yer)):
    cos = cos_matrix(yer[i+1], given)
    print("For Image #" + str(i+1))
    print("\tmean: " + str(cos[0]))
    print("\tminimum: " + str(cos[1]))






#Printing Results with Zhanibek
print("With Zhanibek")

for i in range(len(yer)):
    cos = cos_matrix(yer[i+1], zhanik)
    print("For Image #" + str(i+1))
    print("\tmean: " + str(cos[0]))
    print("\tminimum: " + str(cos[1]))
"""