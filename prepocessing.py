#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import string

def tokenize(word):
    ret = ''
    punc = set(string.punctuation)
    for c in word:
        if c not in punc:
            if c.isnumeric():
                if not ret.endswith('0'):
                    ret += '0'
            else:
                ret += c
        else:
            if not ret.endswith(' '):
                ret += ' '
    return ret.strip()


def main():
    """main"""
    
    # generate relation set and single relation & CVT datasets
    relation = set()
    train_dataset = ['data/EMNLP.train']
    dataset = ['data/EMNLP.dev']
    for train in train_dataset:
        total_line = 0
        fout1 = open(train+'.single', 'w', encoding='utf-8')
        fout2 = open(train+'.cvt', 'w', encoding='utf-8')
        sents = ''
        with open(train, encoding='utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line.startswith('=='):
                    total_line += 1
                    sents = ''
                else:
                    sents += (line+'\n')
                    if line.startswith('<question type'):
                        q_type = line.split('\t')[-1]
                        if q_type == 'single-relation':
                            fout1.write(sents+'==================================================\n')
                        elif q_type == 'cvt':
                            fout2.write(sents+'==================================================\n')
                    elif line.startswith('<logical'):
                        line = line.split('\t')[1]
                        line = line.split()
                        for word in line:
                            if 'mso:' in word:
                                relation.add(word)
        fout1.close()
        fout2.close()

    for test in dataset:
        total_line = 0
        fout1 = open(test+'.single', 'w', encoding='utf-8')
        fout2 = open(test+'.cvt', 'w', encoding='utf-8')
        sents = ''
        with open(test, encoding='utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line.startswith('=='):
                    total_line += 1
                    sents = ''
                else:
                    sents += (line+'\n')
                    if line.startswith('<question type'):
                        q_type = line.split('\t')[-1]
                        if q_type == 'single-relation':
                            fout1.write(sents+'==================================================\n')
                        elif q_type == 'cvt':
                            fout2.write(sents+'==================================================\n')
#                    elif line.startswith('<logical'):
#                        line = line.split('\t')[1]
#                        line = line.split()
#                        for word in line:
#                            if 'mso:' in word:
#                                relation.add(word[4:])
        fout1.close()
        fout2.close()
    
    print('number of relations: ', len(relation))
    relation = sorted(relation)
    fout = open('relation.dict', 'w', encoding='utf-8')
    fout.write('<PAD>\n<UNK>\n')
    for r in relation:
        fout.write(r+'\n')
    fout.close()
    
    # process single-relation
    words = set()
    max_len = 0
    max_len_entity = 0
    train_dataset = ['data/EMNLP.train.single']
    dataset = ['data/EMNLP.dev.single']
    for train in train_dataset:
        total_line = 0
        fout = open(train+'.new', 'w', encoding='utf-8')
        sents = []
        with open(train, encoding='utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line.startswith('=='):
                    total_line += 1
                    fout.write('\t'.join(sents)+'\n')
                    for item in [sents[0], sents[2]]:
                        for word in item.split():
                            words.add(word)
                    max_len_entity = max_len_entity if max_len_entity > len(sents[2].split()) else len(sents[2].split())
                    sents = []
                else:
                    # question
                    if line.startswith('<question id'):
                        sents.append(tokenize(line.split('\t')[1]))
                    #relation
                    elif line.startswith('<logical'):
                        line = line.split('\t')[1]
                        line = line.split()
                        for word in line:
                            if 'mso:' in word:
                                sents.append(word)
                    # entity
                    elif line.startswith('<parameters'):
                        line = line.split('\t')[1]
                        line = line.split()
                        # entity
#                        sents.append(line[0])
                        sents.append(tokenize(line[0]))
                        # location of entity
                        loc = line[-1][1:-1].split(',')
                        question = sents[0].split()
                        question_new = question[:int(loc[0])]
                        question_new.extend(['@'])
                        question_new.extend(question[int(loc[1])+1:])
                        max_len = max_len if max_len > len(question_new) else len(question_new)
                        sents[0] = ' '.join(question_new)
                        
        fout1.close()
        fout2.close()
    print('max_len in training set:', max_len)
    
    
    trigrams = set()
    for word in words:
        word = '#'+word+'#'
        for idx in range(len(word)-2):
            trigrams.add(word[idx:idx+3])
    trigrams = sorted(trigrams)
    print('number of trigrams in training set:', len(trigrams))
    
    fout = open('trigram.dict', 'w', encoding='utf-8')
    fout.write('<PAD>\n<UNK>\n')
    for g in trigrams:
        fout.write(g+'\n')
    fout.close()
    
    for test in dataset:
        total_line = 0
        fout = open(test+'.new', 'w', encoding='utf-8')
        sents = []
        with open(test, encoding='utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line.startswith('=='):
                    total_line += 1
                    fout.write('\t'.join(sents)+'\n')
                    for item in [sents[0], sents[2]]:
                        for word in item.split():
                            words.add(word)
                    max_len_entity = max_len_entity if max_len_entity > len(sents[2].split()) else len(sents[2].split())
                    sents = []
                else:
                    # question
                    if line.startswith('<question id'):
                        sents.append(tokenize(line.split('\t')[1]))
                    # relation
                    elif line.startswith('<logical'):
                        line = line.split('\t')[1]
                        line = line.split()
                        for word in line:
                            if 'mso:' in word:
                                sents.append(word)
                    # entity
                    elif line.startswith('<parameters'):
                        line = line.split('\t')[1]
                        line = line.split()
#                        sents.append(line[0])
                        # entity
                        sents.append(tokenize(line[0]))
                        # location of entity
                        loc = line[-1][1:-1].split(',')
                        question = sents[0].split()
                        question_new = question[:int(loc[0])]
                        question_new.extend(['@'])
                        question_new.extend(question[int(loc[1])+1:])
                        max_len = max_len if max_len > len(question_new) else len(question_new)
                        sents[0] = ' '.join(question_new)
                        
        fout1.close()
        fout2.close()
    print('max_len in all sets:', max_len)
    
    print('number of all words:', len(words))
    words = sorted(words)
    fout = open('word.dict', 'w', encoding='utf-8')
    fout.write('<PAD>\n<UNK>\n')
    for r in words:
        fout.write(r+'\n')
    fout.close()

    word2tri = {}
    for idx, g in enumerate(trigrams):
        word2tri[g] = idx
    
    max_len = 0
    for word in words:
        max_len = max_len if max_len > len(word) else len(word)
    print('max_char:', max_len)
    print('longest_entity:', max_len_entity)
    
    # generate word to trigrams map
    fout = open('word2grams.dict', 'w', encoding='utf-8')
    fout.write('0 ' * max_len + '\n')
    fout.write('0 ' * max_len + '\n')
    for idx, word in enumerate(words):
        word = '#' + word + '#'
        wordmat = np.zeros([max_len], dtype=np.int32)
        for i in range(len(word)-2):
            if word[i:i+3] in trigrams:
                wordmat[i] = word2tri[word[i:i+3]] + 2
            else:
                wordmat[i] = 1
        for i in range(max_len):
            fout.write(str(wordmat[i])+' ')
        fout.write('\n')
    fout.close()
    
    # generate relation to trigrams map
    max_len = 0
    max_char = 0
    for r in relation:
        r = tokenize(r).split()
        max_len = max_len if max_len > len(r) else len(r)
        for w in r:
            max_char = max_char if max_char > len(w) else len(w)
    
    print('max_len for relation:', max_len)
    print('max_char for relation:', max_char)
    relation2grams = np.zeros([len(relation)+2, max_len, max_char], dtype=np.int32)
    for idx, r in enumerate(relation):
        r = tokenize(r).split()
        for j, w in enumerate(r):
            w = '#' + w + '#'
            for i in range(len(w) - 2):
                if w[i:i+3] in trigrams:
                    relation2grams[idx+2, j, i] = word2tri[w[i:i+3]] + 2
                else:
                    relation2grams[idx+2, j, i] = 1
    np.save('relation2grams', relation2grams)
    
    
    
    
    
    
    
    
    
if '__main__' == __name__:
    main()
