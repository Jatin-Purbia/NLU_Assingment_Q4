# Sport vs Politics Text Classification

Binary text classifier using machine learning to distinguish between sports and politics documents.

## Features

- **Algorithms:** Naive Bayes, Logistic Regression, SVM, Random Forest, Decision Tree
- **Feature Extraction:** Bag of Words, TF-IDF, Bigrams, Trigrams
- **Evaluation:** Accuracy, Confusion Matrix, Precision, Recall, F1-Score

## Dataset

- **Total:** 139 documents (111 train, 28 test)
- **Sport:** 69 documents (55 train, 14 test)
- **Politics:** 70 documents (56 train, 14 test)
- **Split:** 80/20 random split

```
data/
├── sport/
│   ├── sport.txt
│   ├── sport_train.txt
│   └── sport_test.txt
└── politics/
    ├── politics.txt
    ├── politics_train.txt
    └── politics_test.txt
```

## Installation

```bash
git clone https://github.com/Jatin-Purbia/NLU_Assingment_Q4.git
cd NLU_Assingment_Q4
pip install -r requirements.txt
```

## Usage

**1. Create train/test split:**
```bash
python split_data.py
```

**2. Train and evaluate models:**
```bash
python evaluation.py
```

**3. Interactive demo:**
```bash
python demo.py
```

**4. Detailed analysis:**
```bash
python detailed_analysis.py
```

## Files

- `split_data.py` - Create 80/20 train/test split
- `feature_extraction.py` - TF-IDF, BoW, N-gram features
- `classifier.py` - ML classifier implementations
- `evaluation.py` - Train and evaluate all models
- `demo.py` - Interactive classification
- `detailed_analysis.py` - Feature analysis

## Results

**100% Accuracy:**
- TF-IDF + Naive Bayes (0.004s train)
- TF-IDF + Logistic Regression (0.003s train)
- TF-IDF + SVM (0.007s train)
- BoW + Naive Bayes (0.001s train)
- BoW + Logistic Regression (0.006s train)
- Bigram + Random Forest (0.115s train)

**96.43% Accuracy:**
- TF-IDF + Random Forest
- BoW + SVM, Random Forest
- Bigram + Naive Bayes, Logistic Regression, SVM

**Best Model:** TF-IDF + Naive Bayes (100% accuracy, fastest)

## Limitations

- Small dataset (139 documents)
- Binary classification only
- Fixed vocabulary (no handling of unknown words)
- Bag-of-words approach (loses context)
- Balanced classes (unrealistic for production)

## Future Improvements

- Larger dataset from news APIs
- Multi-class classification
- Word embeddings (Word2Vec, BERT)
- Deep learning models (LSTM, Transformers)
- Cross-validation for robust evaluation

