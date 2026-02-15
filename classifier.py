# text classifier for sport vs politics classification
# supports multiple ML algorithms

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import pickle

class TextClassifier:
    
    def __init__(self, algorithm='naive_bayes'):
        self.algorithm = algorithm
        self.model = None
        self.is_trained = False
        
        # setup the model based on chosen algorithm
        self._setup_model()
    
    def _setup_model(self):
        # creating the model based on algorithm type
        algo_type = self.algorithm
        
        if algo_type == 'naive_bayes':
            # naive bayes works well for text classification
            self.model = MultinomialNB(alpha=1.0)
            print("Using Naive Bayes classifier")
            
        elif algo_type == 'logistic':
            # good old logistic regression
            self.model = LogisticRegression(max_iter=1000, random_state=42)
            print("Using Logistic Regression")
            
        elif algo_type == 'svm':
            # SVM with linear kernel
            self.model = SVC(kernel='linear', random_state=42)
            print("Using Support Vector Machine")
            
        elif algo_type == 'random_forest':
            # random forest with 100 trees
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            print("Using Random Forest")
            
        elif algo_type == 'decision_tree':
            # decision tree - limited depth to avoid overfitting
            self.model = DecisionTreeClassifier(random_state=42, max_depth=50)
            print("Using Decision Tree")
            
        else:
            # default to naive bayes if algo not recognized
            print(f"Unknown algorithm '{algo_type}', using Naive Bayes instead")
            self.model = MultinomialNB()
    
    def train(self, X_train, y_train):
        # train the model
        print(f"Training {self.algorithm}...")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        print("Training completed!")
        
    def predict(self, X_test):
        # make predictions
        if not self.is_trained:
            print("Warning: Model not trained yet!")
            return None
        
        preds = self.model.predict(X_test)
        return preds
    
    def predict_proba(self, X_test):
        # get probabilities for predictions
        if not self.is_trained:
            return None
        
        try:
            probabilities = self.model.predict_proba(X_test)
            return probabilities
        except:
            # not all models support this
            print("This model doesn't support probability predictions")
            return None
    
    def get_accuracy(self, X_test, y_test):
        # calculate how accurate the predictions are
        predictions = self.predict(X_test)
        
        if predictions is None:
            return 0.0
        
        # count correct predictions
        num_correct = 0
        total = len(y_test)
        
        for i in range(total):
            if predictions[i] == y_test[i]:
                num_correct += 1
        
        acc = (num_correct / total) * 100
        return acc
    
    def cross_validate(self, X, y, cv=5):
        # do cross validation to check generalization
        print(f"Running {cv}-fold cross validation...")
        
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        print(f"Cross validation scores: {cv_scores}")
        avg_score = cv_scores.mean()
        std_score = cv_scores.std()
        print(f"Mean accuracy: {avg_score:.4f} (+/- {std_score:.4f})")
        
        return cv_scores
    
    def save_model(self, filename):
        # save the trained model
        if self.is_trained:
            with open(filename, 'wb') as file:
                pickle.dump(self.model, file)
            print(f"Model saved to {filename}")
        else:
            print("Cannot save: model not trained yet")
    
    def load_model(self, filename):
        # load a saved model
        try:
            with open(filename, 'rb') as file:
                self.model = pickle.load(file)
            self.is_trained = True
            print(f"Model loaded from {filename}")
        except Exception as e:
            print(f"Error loading model: {e}")
