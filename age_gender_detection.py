import pickle
from nltk.corpus.reader.xmldocs import XMLCorpusReader
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import numpy as np
import pickle
from textstat.textstat import textstat
from bs4 import BeautifulSoup
import HTMLParser
from pattern.en import parse
from nltk.tag import pos_tag, map_tag
from nltk.util import bigrams

def create_feature_vect(file_name):
    """
    read from a file a list of words needed by the feature extractor
    :param file_name: path of the file containing the words
    :return:feature list
    """
    with open(file_name) as f:
        feature_vect = f.read().splitlines()

    return feature_vect

def gender_feature(text, feature_vector):
    """
    Extract the gender features
    :param text:
    :param feature_vector: contains a bag of words
    :return: a dictionary which contains the feature and its computed value
    """

    #sentence length and vocab features
    tokens = word_tokenize(text.lower())
    sentences = sent_tokenize(text.lower())

    words_per_sent = np.asarray([len(word_tokenize(s)) for s in sentences])
    #print words_per_sent
    #print words_per_sent.std()

    #bag_of_word features
    bag_feature = {}
    for word in feature_vector:
        bag_feature['contains(%s)' % word] = (word in set(tokens))

    #POS tagging features
#    POS_tag = ['ADJ', 'ADV', 'DET', 'NOUN', 'PRT', 'VERB', '.']
 #   tagged_word = parse(text, chunks=False, tagset='UNIVERSAL').split()
  #  simplified_tagged_word = [(tag[0], map_tag('en-ptb', 'universal', tag[1])) for s in tagged_word for tag in s]
   # freq_POS = nltk.FreqDist(tag[1] for tag in simplified_tagged_word if tag[1] in POS_tag)

    d = dict({'sentence_length_variation': words_per_sent.std()}, **bag_feature)

    #return dict(d, **freq_POS)
    return d

def age_feature(text, feature_vector):
    """
    Extract age features
    :param text:
    :param feature_vector: contains a bag of words
    :return:a dictionary which contains the feature and its computed value
    """
    tokens = word_tokenize(text.lower())
    features={}
    for word in feature_vector:
        features['contains(%s)' % word] = (word in set(tokens))

    #print features
    d=dict(features, **dict({'FRE': textstat.flesch_reading_ease(text), 'FKGL': textstat.flesch_kincaid_grade(text)}))
    return d





def feature_apply(feature_extractor, feature_vector, attribute, number_of_file):
    """
    Extract features from each document
    :param feature_extractor: function that extract features
    :param feature_vector: contains a list of features
    :param attribute: indicate if the process for gender or age feature extraction
    :param number_of_file: number of document to be processed
    :return:vector that contain the extracted features
    """
    corpus_root = '/root/Downloads/TextMining/pan13-author-profiling-training-corpus-2013-01-09/en'
    #corpus_root = '/root/Downloads/TextMining/pan13-author-profiling-training-corpus-2013-01-09/meTets'
    newcorpus = XMLCorpusReader(corpus_root, '.*')
    i=0
    feature_set = []
    doc_list = newcorpus.fileids()
    print len(doc_list)

    for doc in doc_list[:number_of_file]:
        i+=1
        if i%50==0:
            print i
        doc = newcorpus.xml(doc)
        number_of_conversation=int(doc[0].attrib["count"])
        #print(doc[0].attrib["count"])
        txt = " ".join([doc[0][j].text for j in range(number_of_conversation) if doc[0][j].text is not None])
        #print txt
        if textstat.sentence_count(txt) != 0:
            feature_set.append((feature_extractor(txt, feature_vector), doc.attrib[attribute]))

    return feature_set

def generate_classifier(feature_set, n_sample):
    """
    Divide the feature vector to a training set and development set
    Train a NaiveBayes classifier with the training set and evaluate the accuracy with development set.
    :param feature_set:
    :param n_sample: number of document
    :return:the trained classifier
    """

    #train_len = int(len(feature_set)*0.9)
    #train_set, dev_set = feature_set[:train_len], feature_set[train_len:]


    #print len(feature_set)
    #train_set= feature_set
    classifier = nltk.NaiveBayesClassifier.train(feature_set)
    #save_classifier(classifier, "Naive", len(feature_set[0][0]), n_sample)
  #  print  (dev_set)
   # print nltk.classify.accuracy(classifier, dev_set)
    #classifier.show_most_informative_features(5)
    return classifier

def save_classifier(classifier,  n_feat, n_sample):
    """
    Save the classifier in a pickle file with a noun indicating date, number of documents
    and number of features used in to train the classifier
    :param classifier:
    :param n_feat: number of features
    :param n_sample: number of documents
    :return:
    """
    file_name = 'feat'+str(n_feat)+'_sample'+str(n_sample)+'.pickle'
    f = open(file_name, 'wb')
    pickle.dump(classifier, f)
    f.close()


#******************************************Testing functions

#test create_feature
#file contain specific word determing age
age_words_file='/root/Downloads/TextMining/age_words.txt'
gender_words_file='/root/Downloads/TextMining/gender_words.txt'
my_gender_words_file='/root/Downloads/TextMining/my_gender_words.txt'
my_age_words_file='/root/Downloads/TextMining/my_age_words.txt'

#age_words='boring student bored college'.split()
age_words=create_feature_vect(age_words_file)
gender_word=create_feature_vect(gender_words_file)
my_gender_word=create_feature_vect(my_gender_words_file)
#print age_words
#print gender_word

#Example of text
#text="hello this an awesome text example is just a boring dumb text for testing purpose"

#test age_feature
#extract age feature from a text
#print age_feature(text,age_words)

#Test feature apply

#feature_set=feature_apply(age_feature,age_words,'age_group',236600)
feature_set=feature_apply(gender_feature,my_gender_word,'gender',236600)
#my_feature_set=feature_apply(gender_feature,gender_word,'gender',1)
#print feature_set

#Test classifier

classifier=generate_classifier(feature_set,1)
save_classifier(classifier,2,236600)
#classifier2=generate_classifier(my_feature_set,10)
#classifier = nltk.NaiveBayesClassifier.train(feature_set)
#classifier.show_most_informative_features(50)
#test='pink'
#test='college'
#print test +' was written by a '
#print classifier.classify(gender_feature(test, gender_word))
#print classifier.classify(age_feature(test, age_words))
#save_classifier(classifier, "Naive", len(feature_set[0][0]), 200000)