# demo script - test the classifier on whatever text you type

import os
import numpy as np
from feature_extraction import FeatureExtractor
from classifier import TextClassifier

def load_training_data():
    # load training data from files
    docs = []
    labels = []
    
    # sport data
    sport_path = 'data/sport/sport_train.txt'
    if os.path.exists(sport_path):
        file = open(sport_path, 'r', encoding='utf-8')
        for line in file:
            text = line.strip()
            if text:
                docs.append(text)
                labels.append(0)  # 0 for sport
        file.close()
    
    # politics data
    politics_path = 'data/politics/politics_train.txt'
    if os.path.exists(politics_path):
        file = open(politics_path, 'r', encoding='utf-8')
        for line in file:
            text = line.strip()
            if text:
                docs.append(text)
                labels.append(1)  # 1 for politics
        file.close()
    
    return docs, labels

def predict_text(text, classifier, feature_extractor):
    # predict what category the text belongs to
    
    # turn text into features
    X = feature_extractor.transform([text])
    
    # check if we recognized any words
    total_features = X.sum()
    
    # make prediction
    pred = classifier.predict(X)[0]
    
    # try to get probabilities
    try:
        probabilities = classifier.predict_proba(X)[0]
        return pred, probabilities, total_features
    except:
        return pred, None, total_features

def main():
    print("="*60)
    print("Sport vs Politics Text Classifier Demo")
    print("="*60)
    
    # make sure data exists
    if not os.path.exists('data'):
        print("\nError: Data folder not found!")
        print("Run split_data.py first to create the dataset.")
        return
    
    # load training docs
    print("\nLoading training data...")
    docs, labels = load_training_data()
    
    if len(docs) == 0:
        print("No training data found!")
        return
    
    print(f"Loaded {len(docs)} training documents")
    
    # create feature extractor - tfidf works best
    print("\nExtracting features...")
    feature_extractor = FeatureExtractor(method='tfidf')
    extractor = FeatureExtractor(method='tfidf')
    X_train = extractor.fit_transform(docs)
    y_train = np.array(labels)
    
    # train classifier - logistic regression is fast and works well
    print("\nTraining classifier...")
    clf = TextClassifier(algorithm='logistic')
    clf.train(X_train, y_train)
    
    print("\n" + "="*60)
    print("Model trained! Ready to classify your text.")
    print("="*60)
    
    # main loop - keep asking for text to classify
    while True:
        print("\n" + "-"*60)
        print("Enter text to classify (or 'quit' to exit):")
        print("-"*60)
        
        text_input = input("> ")
        
        # check if user wants to quit
        if text_input.lower() in ['quit', 'exit', 'q']:
            print("\nThanks for using the classifier!")
            break
        
        if len(text_input.strip()) == 0:
            print("Please enter some text!")
            continue
        
        # classify the text
        pred, probs, feat_sum = predict_text(text_input, clf, extractor)
        
        # warn if no words recognized
        if feat_sum == 0:
            print("\nWARNING: No words from your text were seen during training!")
            print("The prediction might not be reliable.")
        
        # show prediction
        if pred == 0:
            category = "SPORT"
        else:
            category = "POLITICS"
        
        print(f"\nPrediction: {category}")
        
        # show probabilities if available
        if probs is not None:
            sport_conf = probs[0] * 100
            politics_conf = probs[1] * 100
            print(f"Confidence: Sport={sport_conf:.2f}%, Politics={politics_conf:.2f}%")
            
            # check if confidence is low
            diff = abs(probs[0] - probs[1])
            if diff < 0.05:  # less than 5% difference
                print("\nLow confidence - text might have unknown words.")
    
    # show some example texts
    print("\n" + "="*60)
    print("Example texts you can try:")
    print("="*60)
    print("\n1. The basketball team won the championship game yesterday.")
    print("2. The government announced new legislation for tax reform.")
    print("3. Athletes are preparing for the Olympic games this summer.")
    print("4. The parliament debated healthcare policies for hours.")

if __name__ == "__main__":
    main()
