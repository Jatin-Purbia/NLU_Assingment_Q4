# feature extraction - converts text to numbers for ML

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np

class FeatureExtractor:
    
    def __init__(self, method='tfidf'):
        # can use bow, tfidf, bigram, trigram
        self.method = method
        self.vectorizer = None
        
    def fit_transform(self, texts):
        # convert text documents into feature vectors
        # this is where we turn words into numbers
        
        method_type = self.method
        
        # simple word count
        if method_type == 'bow':
            self.vectorizer = CountVectorizer(max_features=1000, stop_words='english')
            X = self.vectorizer.fit_transform(texts)
            print(f"BoW features created with shape: {X.shape}")
            
        # tf-idf gives more weight to important words
        elif method_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            X = self.vectorizer.fit_transform(texts)
            print(f"TF-IDF features created with shape: {X.shape}")
            
        # bigrams capture word pairs
        elif method_type == 'bigram':
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),  # both single words and pairs
                max_features=1500,
                stop_words='english'
            )
            X = self.vectorizer.fit_transform(texts)
            print(f"Bigram features created with shape: {X.shape}")
            
        # trigrams use 3-word sequences
        elif method_type == 'trigram':
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=2000,
                stop_words='english'
            )
            X = self.vectorizer.fit_transform(texts)
            print(f"Trigram features created with shape: {X.shape}")
            
        else:
            # if unknown method, just use tfidf
            print(f"Unknown method '{method_type}', defaulting to TF-IDF")
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            X = self.vectorizer.fit_transform(texts)
        
        return X
    
    def transform(self, texts):
        # transform new text using already fitted vectorizer
        if self.vectorizer is None:
            print("Error: Need to fit vectorizer first!")
            return None
        
        X = self.vectorizer.transform(texts)
        return X
    
    def get_feature_names(self):
        # get the actual words/ngrams used as features
        if self.vectorizer is not None:
            try:
                # newer sklearn version
                return self.vectorizer.get_feature_names_out()
            except:
                # older version
                return self.vectorizer.get_feature_names()
        return None
