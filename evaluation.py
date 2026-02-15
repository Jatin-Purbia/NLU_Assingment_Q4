# evaluation script - trains and compares different classifiers

import os
from feature_extraction import FeatureExtractor
from classifier import TextClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

def load_documents(folder_path):
    # load training documents from text files
    # each line in the file is one document
    
    docs = []
    labels = []
    
    # first load sport documents
    sport_path = os.path.join(folder_path, 'sport', 'sport_train.txt')
    if os.path.exists(sport_path):
        f = open(sport_path, 'r', encoding='utf-8')
        lines = f.readlines()
        f.close()
        
        for line in lines:
            text = line.strip()
            if text:  # ignore empty lines
                docs.append(text)
                labels.append(0)  # 0 = sport
    
    # then load politics documents
    politics_path = os.path.join(folder_path, 'politics', 'politics_train.txt')
    if os.path.exists(politics_path):
        f = open(politics_path, 'r', encoding='utf-8')
        lines = f.readlines()
        f.close()
        
        for line in lines:
            text = line.strip()
            if text:
                docs.append(text)
                labels.append(1)  # 1 = politics
    
    print(f"Loaded {len(docs)} documents")
    print(f"Sport documents: {labels.count(0)}")
    print(f"Politics documents: {labels.count(1)}")
    
    return docs, labels

def load_test_documents(folder_path):
    # load test documents
    docs = []
    labels = []
    
    # load sport test data
    sport_test_path = folder_path + '/sport/sport_test.txt'
    sport_test_path = sport_test_path.replace('/', os.sep)
    
    with open(sport_test_path, 'r', encoding='utf-8') as file:
        content = file.read()
        lines = content.split('\n')
        for l in lines:
            if len(l.strip()) > 0:
                docs.append(l.strip())
                labels.append(0)
    
    # load politics test data
    politics_test_path = folder_path + '/politics/politics_test.txt'
    politics_test_path = politics_test_path.replace('/', os.sep)
    
    with open(politics_test_path, 'r', encoding='utf-8') as file:
        content = file.read()
        lines = content.split('\n')
        for l in lines:
            if len(l.strip()) > 0:
                docs.append(l.strip())
                labels.append(1)
    
    print(f"Loaded {len(docs)} test documents")
    return docs, labels

def plot_confusion_matrix(cm, classifier_name, save_path='results'):
    # make a confusion matrix plot
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Sport', 'Politics'],
                yticklabels=['Sport', 'Politics'])
    plt.title(f'Confusion Matrix - {classifier_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # make sure results folder exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # save figure
    save_file = os.path.join(save_path, f'confusion_matrix_{classifier_name}.png')
    plt.savefig(save_file)
    print(f"Saved confusion matrix to {save_file}")
    plt.close()

def compare_classifiers():
    # main function - compare all classifiers with different features
    
    print("="*50)
    print("Text Classification: Sport vs Politics")
    print("="*50)
    
    # load the training data
    print("\nLoading training data...")
    train_docs, train_labels = load_documents('data')
    
    # load test data
    print("\nLoading test data...")
    test_docs, test_labels = load_test_documents('data')
    
    # try these feature methods
    feature_methods = ['tfidf', 'bow', 'bigram']
    
    # try these classifiers
    classifiers_to_try = ['naive_bayes', 'logistic', 'svm', 'random_forest']
    
    all_results = {}
    
    # loop through each feature method
    idx = 0
    while idx < len(feature_methods):
        method = feature_methods[idx]
        
        print(f"\n{'='*50}")
        print(f"Feature Method: {method.upper()}")
        print(f"{'='*50}")
        
        # extract features from training data
        print("\nExtracting features from training data...")
        extractor = FeatureExtractor(method=method)
        X_train = extractor.fit_transform(train_docs)
        y_train = np.array(train_labels)
        
        # extract features from test data
        print("Extracting features from test data...")
        X_test = extractor.transform(test_docs)
        y_test = np.array(test_labels)
        
        idx += 1
        
        # try different classifiers
        for clf_name in classifiers_to_try:
            print(f"\n{'-'*40}")
            print(f"Classifier: {clf_name.upper()}")
            print(f"{'-'*40}")
            
            # create classifier
            clf = TextClassifier(algorithm=clf_name)
            
            # train and time it
            t1 = time.time()
            clf.train(X_train, y_train)
            t2 = time.time()
            train_time = t2 - t1
            
            # predict and time it
            t1 = time.time()
            preds = clf.predict(X_test)
            t2 = time.time()
            pred_time = t2 - t1
            
            # calculate accuracy
            acc = clf.get_accuracy(X_test, y_test)
            
            # print classification report
            print("\nClassification Report:")
            report = classification_report(y_test, preds, 
                                          target_names=['Sport', 'Politics'])
            print(report)
            
            # make confusion matrix
            cm = confusion_matrix(y_test, preds)
            plot_confusion_matrix(cm, f"{method}_{clf_name}")
            
            # save results
            result_key = f"{method}_{clf_name}"
            all_results[result_key] = {
                'accuracy': acc,
                'training_time': train_time,
                'prediction_time': pred_time,
                'confusion_matrix': cm
            }
            
            print(f"Accuracy: {acc:.2f}%")
            print(f"Training Time: {train_time:.4f} seconds")
            print(f"Prediction Time: {pred_time:.4f} seconds")
    
    # make comparison plots
    print("\n" + "="*50)
    print("Creating comparison plots...")
    create_comparison_plots(all_results)
    
    # save all results
    save_results_to_file(all_results)
    
    print("\n" + "="*50)
    print("Evaluation completed!")
    print("="*50)
    print("="*50)
    
    return results

def create_comparison_plots(results):
    """
    Create bar charts comparing different models
    """
    # extract data for plotting
    names = []
    accuracies = []
    
    for key, value in results.items():
        names.append(key)
        accuracies.append(value['accuracy'])
    
    # create bar plot
    plt.figure(figsize=(14, 6))
    bars = plt.bar(range(len(names)), accuracies, color='skyblue', edgecolor='navy')
    plt.xlabel('Classifier Configuration', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Comparison of Different Classifiers', fontsize=14, fontweight='bold')
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    
    # add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if not os.path.exists('results'):
        os.makedirs('results')
    
    plt.savefig('results/classifier_comparison.png', dpi=300)
    print("Saved comparison plot to results/classifier_comparison.png")
    plt.close()

def save_results_to_file(results):
    """
    Save numerical results to a text file
    """
    if not os.path.exists('results'):
        os.makedirs('results')
    
    with open('results/comparison_results.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("CLASSIFIER COMPARISON RESULTS\n")
        f.write("="*60 + "\n\n")
        
        # find best performer
        best_config = max(results.items(), key=lambda x: x[1]['accuracy'])
        
        f.write(f"Best Configuration: {best_config[0]}\n")
        f.write(f"Best Accuracy: {best_config[1]['accuracy']:.2f}%\n\n")
        
        f.write("="*60 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("="*60 + "\n\n")
        
        for config, metrics in results.items():
            f.write(f"Configuration: {config}\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.2f}%\n")
            f.write(f"  Training Time: {metrics['training_time']:.4f} seconds\n")
            f.write(f"  Prediction Time: {metrics['prediction_time']:.4f} seconds\n")
            f.write(f"  Confusion Matrix:\n")
            f.write(f"    {metrics['confusion_matrix'][0]}\n")
            f.write(f"    {metrics['confusion_matrix'][1]}\n")
            f.write("\n")
    
    print("Saved detailed results to results/comparison_results.txt")

if __name__ == "__main__":
    # run the comparison
    results = compare_classifiers()
