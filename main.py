import os
import caption
import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from scipy.misc import imread, imresize
from PIL import Image
from encoder import Encoder
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

myenc = Encoder()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def directory_to_sentence_matrix(directory):
    sentence_dict = {}
    for j in range(1,7):
        filename = directory + "/" + str(j) + ".jpg"
        checkpoint = torch.load('/home/yerlan/HackNU/a-PyTorch-Tutorial-to-Image-Captioning/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar')
        decoder = checkpoint['decoder']
        decoder = decoder.to(device)
        decoder.eval()
        encoder = checkpoint['encoder']
        encoder = encoder.to(device)
        encoder.eval()

        # Load word map (word2ix)
        with open('/home/yerlan/HackNU/a-PyTorch-Tutorial-to-Image-Captioning/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json' , 'r') as t:
            word_map = json.load(t)
        rev_word_map = {v: k for k, v in word_map.items()}  # ix2word


        sentence_array = []
        # Encode, decode with 0attention and beam search
        for i in range(1, 6):
            seq, alphas = caption.caption_image_beam_search(encoder, decoder, filename, word_map, i)
            alphas = torch.FloatTensor(alphas)

            # Visualize caption and attention of best sequence
            sentence_array.append(caption.return_sentence(filename, seq, alphas, rev_word_map))
        sentence_dict[j] = sentence_array

    return sentence_dict

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


#number of files n
n = 2
path = "/home/yerlan/HackNU/images/"
sentence_matrix = []
#loop iterating over directories
for i in range(1,n):
    directory = path + str(i)
    # below is for iterating over images in the folder, please uncomment to activate
    sentence_dictionary = directory_to_sentence_matrix(directory)


    sent_dir = directory + "/input"
    yer = sentence_dictionary
    given = filter_input1(sent_dir)
    zhanik = filter_input2(sent_dir)

    #Printing Results without Zhanibek
    print("Without Zhanibek")

    for k in range(len(yer)):
        cos = cos_matrix(yer[k+1], given)
        print("For Image #" + str(k+1))
        print("\tmean: " + str(cos[0]))
        print("\tminimum: " + str(cos[1]))


    #Printing Results with Zhanibek
    print("With Zhanibek")

    for j in range(len(yer)):
        cos = cos_matrix(yer[j+1], zhanik)
        print("For Image #" + str(j+1))
        print("\tmean: " + str(cos[0]))
        print("\tminimum: " + str(cos[1]))





