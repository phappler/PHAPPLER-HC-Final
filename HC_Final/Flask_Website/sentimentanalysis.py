from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def analyzesentiment(document, rsplt):
    filename = "G:\My Drive\Hacking Chinese\HC_Final\Flask_Website\Texts\\" + document
    with open(filename, 'r', encoding='utf8') as file:
        rawtext = file.read()
    #seperates Gutenberg's informational sections from usable text
    textbody = re.split(r'\*\*\*.*\*\*\*', rawtext)[1]

    sections = re.split(rsplt, textbody)

    n_instances = 100
    subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
    obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]

    train_subj_docs = subj_docs[:80]
    test_subj_docs = subj_docs[80:100]
    train_obj_docs = obj_docs[:80]
    test_obj_docs = obj_docs[80:100]
    training_docs = train_subj_docs+train_obj_docs
    testing_docs = test_subj_docs+test_obj_docs

    sentim_analyzer = SentimentAnalyzer()
    all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])
    
    unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
    sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)

    training_set = sentim_analyzer.apply_features(training_docs)
    test_set = sentim_analyzer.apply_features(testing_docs)

    trainer = NaiveBayesClassifier.train
    classifier = sentim_analyzer.train(trainer, training_set)
    for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
        print('{0}: {1}'.format(key, value))
    negsent = []
    possent = []
    for n in range(len(sections)):
        section = sections[n]
        sid = SentimentIntensityAnalyzer()
        ss = sid.polarity_scores(section)
        sss = sorted(ss)

        negsent.append((ss[sss[1]], n))
        possent.append((ss[sss[3]], n))
    negsent.sort(reverse=True)
    possent.sort(reverse=True)

    mydict = {}
    mydict["negsent"] = negsent #sentiment val, section number (sorted by value descending)
    mydict["possent"] = possent #^
    mydict["sections"] = sections #sections in order as they appear

    return mydict

#'''