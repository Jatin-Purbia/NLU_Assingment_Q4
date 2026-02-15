# script to split data into 80% training and 20% testing
# reads politics.txt and sport.txt, then creates train/test files

import os
import random

def read_and_split_file(file_path, train_ratio=0.8):
    # read file and split paragraphs into train/test sets
    
    f = open(file_path, 'r', encoding='utf-8')
    content = f.read()
    f.close()
    
    # split by newlines to get individual paragraphs
    paragraphs = content.split('\n')
    # remove empty lines
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    print(f"Total paragraphs in {file_path}: {len(paragraphs)}")
    
    # shuffle randomly (with seed for reproducibility)
    random.seed(42)
    random.shuffle(paragraphs)
    
    # calculate where to split
    num_train = int(len(paragraphs) * train_ratio)
    
    train_paras = paragraphs[:num_train]
    test_paras = paragraphs[num_train:]
    
    print(f"  Train: {len(train_paras)} documents")
    print(f"  Test: {len(test_paras)} documents")
    
    return train_paras, test_paras

def write_data_file(data, file_path):
    # write data to file (one document per line)
    f = open(file_path, 'w', encoding='utf-8')
    for doc in data:
        f.write(doc + '\n')
    f.close()
    print(f"Written to {file_path}")

def main():
    print("="*60)
    print("Creating 80/20 Train/Test Split")
    print("="*60)
    
    # split sport data
    print("\nProcessing sport data...")
    sport_train, sport_test = read_and_split_file('data/sport/sport.txt')
    
    # split politics data
    print("\nProcessing politics data...")
    politics_train, politics_test = read_and_split_file('data/politics/politics.txt')
    
    # write training files
    print("\n" + "="*60)
    print("Writing train files...")
    print("="*60)
    write_data_file(sport_train, 'data/sport/sport_train.txt')
    write_data_file(politics_train, 'data/politics/politics_train.txt')
    
    # write test files
    print("\n" + "="*60)
    print("Writing test files...")
    print("="*60)
    write_data_file(sport_test, 'data/sport/sport_test.txt')
    write_data_file(politics_test, 'data/politics/politics_test.txt')
    
    # summary
    print("\n" + "="*60)
    print("Split complete!")
    print("="*60)
    total_train = len(sport_train) + len(politics_train)
    total_test = len(sport_test) + len(politics_test)
    print(f"\nTotal training documents: {total_train}")
    print(f"  - Sport: {len(sport_train)}")
    print(f"  - Politics: {len(politics_train)}")
    print(f"\nTotal test documents: {total_test}")
    print(f"  - Sport: {len(sport_test)}")
    print(f"  - Politics: {len(politics_test)}")

if __name__ == '__main__':
    main()
