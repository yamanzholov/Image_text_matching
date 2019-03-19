import tensorflow as tf
import tensorflow_hub as hub
from nltk.tokenize import sent_tokenize
import time
from mtranslate import translate
from tqdm import tqdm
import numpy as np
import scipy.spatial

class Encoder:
    def __init__(self, ):
        g = tf.Graph()
        with g.as_default():
            # We will be feeding 1D tensors of text into the graph.
            self.text_input = tf.placeholder(dtype=tf.string, shape=[None])
            embed = hub.Module("/home/yerlan/HackNU/a-PyTorch-Tutorial-to-Image-Captioning/module_useT")
            self.embedded_text = embed(self.text_input)
            init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
        g.finalize()
        self.session = tf.Session(graph=g)
        self.session.run(init_op)
    def encode(self, text):
        if not isinstance(text, list):
            text = [text]
        result = self.session.run(self.embedded_text, feed_dict={self.text_input: text})
        return result
    
    def cosine_similarity(self, matrix, v):
        return scipy.spatial.distance.cdist(matrix, v, 'cosine')
    
    def closest_vector(self,v, v_question):
        scores = self.cosine_similarity(v, v_question)
        index = np.unravel_index(scores.argmin(), scores.shape)
        if v_question.shape[0]>1:
            return index[-1],scores[index]
        else:
            return index[0],scores[index]
