# analysis script - look at what features are important

import os
import numpy as np
from feature_extraction import FeatureExtractor
from classifier import TextClassifier

def load_training_data():
    # load training data from files
    docs = []
    labels = []
    
    # load sport documents
    sport_path = 'data/sport/sport_train.txt'
    with open(sport_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            text = line.strip()
            if text:
                docs.append(text)
                labels.append(0)
    
    # load politics documents
    politics_path = 'data/politics/politics_train.txt'
    with open(politics_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            text = line.strip()
            if text:
                docs.append(text)
                labels.append(1)
    
    return docs, labels

def analyze_vocabulary():
    # analyze which words are most important for each category
    
    print("="*60)
    print("VOCABULARY ANALYSIS")
    print("="*60)
    
    # load data
    docs, labels = load_training_data()
    
    # extract tfidf features
    print("\nExtracting TF-IDF features...")
    extractor = FeatureExtractor(method='tfidf')
    X = extractor.fit_transform(docs)
    
    # get feature names (words)
    words = extractor.get_feature_names()
    
    # separate sport and politics documents
    labels_arr = np.array(labels)
    sport_idx = labels_arr == 0
    politics_idx = labels_arr == 1
    
    # convert to dense for easier calculations
    X_array = X.toarray()
    
    # calculate average tfidf for each word in each class
    sport_scores = X_array[sport_idx].mean(axis=0)
    politics_scores = X_array[politics_idx].mean(axis=0)
    
    # find top words for sport
    print("\n" + "-"*60)
    print("TOP 20 WORDS FOR SPORT:")
    print("-"*60)
    top_sport = sport_scores.argsort()[-20:][::-1]
    for i in top_sport:
        word_name = words[i]
        word_score = sport_scores[i]
        print(f"{word_name:20s} : {word_score:.4f}")
    
    # find top words for politics
    print("\n" + "-"*60)
    print("TOP 20 WORDS FOR POLITICS:")
    print("-"*60)
    top_politics = politics_scores.argsort()[-20:][::-1]
    for i in top_politics:
        word_name = words[i]
        word_score = politics_scores[i]
        print(f"{word_name:20s} : {word_score:.4f}")
    
    # find most discriminative words
    print("\n" + "-"*60)
    print("MOST DISCRIMINATIVE WORDS:")
    print("-"*60)
    print("\nWords that differ most between categories")
    
    # calculate difference between categories
    difference = sport_scores - politics_scores
    
    print("\nMost indicative of SPORT:")
    sport_disc = difference.argsort()[-15:][::-1]
    for i in sport_disc:
        w = words[i]
        diff_val = difference[i]
        print(f"{w:20s} : difference = {diff_val:.4f}")
    
    print("\nMost indicative of POLITICS:")
    politics_disc = difference.argsort()[:15]
    for i in politics_disc:
        w = words[i]
        diff_val = -difference[i]  # negative to show positive value
        print(f"{w:20s} : difference = {diff_val:.4f}")

def analyze_logistic_regression_weights():
    # look at what logistic regression learned
    
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION FEATURE WEIGHTS")
    print("="*60)
    
    # load data
    docs, labels = load_training_data()
    
    # get features
    extractor = FeatureExtractor(method='tfidf')
    X = extractor.fit_transform(docs)
    y = np.array(labels)
    
    # train logistic regression
    print("\nTraining Logistic Regression...")
    clf = TextClassifier(algorithm='logistic')
    clf.train(X, y)
    
    # get the weights
    words = extractor.get_feature_names()
    coefficients = clf.model.coef_[0]
    
    print("\n" + "-"*60)
    print("TOP 15 FEATURES FOR SPORT (negative weights):")
    print("-"*60)
    sport_idx = coefficients.argsort()[:15]
    for i in sport_idx:
        w = words[i]
        coef = coefficients[i]
        print(f"{w:20s} : {coef:+.4f}")
    
    
    print("\n" + "-"*60)
    print("TOP 15 FEATURES FOR POLITICS (positive weights):")
    print("-"*60)
    politics_idx = coefficients.argsort()[-15:][::-1]
    for i in politics_idx:
        w = words[i]
        coef = coefficients[i]
        print(f"{w:20s} : {coef:+.4f}")

def dataset_statistics():
    # show some basic stats about the dataset
    
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    # load documents
    docs, labels = load_training_data()
    
    # count words in each document
    lengths = []
    for doc in docs:
        word_list = doc.split()
        lengths.append(len(word_list))
    
    lengths_array = np.array(lengths)
    
    print(f"\nTotal training documents: {len(docs)}")
    print(f"Sport documents: {labels.count(0)}")
    print(f"Politics documents: {labels.count(1)}")
    
    print(f"\nDocument length statistics:")
    print(f"  Average words per document: {lengths_array.mean():.2f}")
    print(f"  Minimum words: {lengths_array.min()}")
    print(f"  Maximum words: {lengths_array.max()}")
    print(f"  Standard deviation: {lengths_array.std():.2f}")
    
    # count unique words  
    unique_words = set()
    for doc in docs:
        word_list = doc.lower().split()
        unique_words.update(word_list)
    
    print(f"\nTotal unique words (raw): {len(unique_words)}")
    
    # total word count
    total = sum(lengths)
    print(f"Total words in corpus: {total}")
    
    # compare sport vs politics lengths
    sport_lens = []
    politics_lens = []
    for i in range(len(lengths)):
        if labels[i] == 0:
            sport_lens.append(lengths[i])
        else:
            politics_lens.append(lengths[i])
    
    print(f"\nAverage words in Sport documents: {np.mean(sport_lens):.2f}")
    print(f"Average words in Politics documents: {np.mean(politics_lens):.2f}")

def main():
    # run all analyses
    
    print("="*60)
    print("DETAILED ANALYSIS")
    print("="*60)
    
    # check if data folder exists
    if not os.path.exists('data'):
        print("\nError: Data folder not found!")
        print("Run split_data.py first.")
        return
    
    # run the different analyses
    dataset_statistics()
    analyze_vocabulary()
    analyze_logistic_regression_weights()
    
    print("\n" + "="*60)
    print("Analysis completed!")
    print("="*60)

if __name__ == "__main__":
    main()
