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
                                relation.add(word.split('mso:')[-1])
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
#    fout.write('<PAD>\n<UNK>\n')
    for r in relation:
        fout.write(r+'\n')
    fout.close()
    
    # process single-relation
    words = set()
    relation = set()
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
                    for item in [sents[1], sents[3]]:
                        for word in item.split():
                            words.add(word)
                    max_len_entity = max_len_entity if max_len_entity > len(sents[3].split()) else len(sents[3].split())
                    sents = []
                else:
                    # question
                    if line.startswith('<question id'):
                        sents.append(line.split('\t')[0].split('=')[-1][:-1])
                        sents.append(tokenize(line.split('\t')[1]))
                    #relation
                    elif line.startswith('<logical'):
                        line = line.split('\t')[1]
                        line = line.split()
                        for word in line:
                            if 'mso:' in word:
                                relation.add(word.split('mso:')[-1])
                                sents.append(word.split('mso:')[-1])
                    # entity
                    elif line.startswith('<parameters'):
                        line = line.split('\t')[1]
                        line = line.split()
                        # entity
#                        sents.append(line[0])
                        sents.append(tokenize(line[0]))
                        # location of entity
                        loc = line[-1][1:-1].split(',')
                        question = sents[1].split()
                        question_new = question[:int(loc[0])]
                        question_new.extend(['@'])
                        question_new.extend(question[int(loc[1])+1:])
                        max_len = max_len if max_len > len(question_new) else len(question_new)
                        sents[1] = ' '.join(question_new)
                        
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
                    for item in [sents[1], sents[3]]:
                        for word in item.split():
                            words.add(word)
                    max_len_entity = max_len_entity if max_len_entity > len(sents[3].split()) else len(sents[3].split())
                    sents = []
                else:
                    # question
                    if line.startswith('<question id'):
                        sents.append(line.split('\t')[0].split('=')[-1][:-1])
                        sents.append(tokenize(line.split('\t')[1]))
                    # relation
                    elif line.startswith('<logical'):
                        line = line.split('\t')[1]
                        line = line.split()
                        for word in line:
                            if 'mso:' in word:
                                relation.add(word.split('mso:')[-1])
                                sents.append(word.split('mso:')[-1])
                    # entity
                    elif line.startswith('<parameters'):
                        line = line.split('\t')[1]
                        line = line.split()
#                        sents.append(line[0])
                        # entity
                        sents.append(tokenize(line[0]))
                        # location of entity
                        loc = line[-1][1:-1].split(',')
                        question = sents[1].split()
                        question_new = question[:int(loc[0])]
                        question_new.extend(['@'])
                        question_new.extend(question[int(loc[1])+1:])
                        max_len = max_len if max_len > len(question_new) else len(question_new)
                        sents[1] = ' '.join(question_new)

        fout1.close()
        fout2.close()
    print('max_len in all sets:', max_len)
    
    print('number of relations in all sets:', len(relation))
    relation = sorted(relation)
    fout = open('relation.single.dict', 'w', encoding='utf-8')
#    fout.write('<PAD>\n<UNK>\n')
    for r in relation:
        fout.write(r+'\n')
    fout.close()
    
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
    word2id = {}
    for idx, w in enumerate(words):
        word2id[w] = idx
    
    max_len = 0
    for word in words:
        max_len = max_len if max_len > len(word) else len(word)
    print('max_char:', max_len)
    print('longest_entity:', max_len_entity)
    
    word2grams = np.zeros([len(words)+2, max_len], dtype=np.int32)
    for idx, word in enumerate(words):
        word = '#' + word + '#'
        for i in range(len(word)-2):
            if word[i:i+3] in trigrams:
                word2grams[idx+2, i] = word2tri[word[i:i+3]] + 2
            else:
                word2grams[idx+2, i] = 1
    np.save('word2grams', word2grams)
    
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

    relation2grams = np.zeros([len(relation), max_len, max_char], dtype=np.int32)
    relation2words = np.zeros([len(relation), max_len], dtype=np.int32)
    for idx, r in enumerate(relation):
        r = tokenize(r).split()
        for j, w in enumerate(r):
            if w in word2id:
                relation2words[idx, j] = word2id[w] + 2
            else:
                relation2words[idx, j] = 1
            w = '#' + w + '#'
            for i in range(len(w) - 2):
                if w[i:i+3] in trigrams:
                    relation2grams[idx, j, i] = word2tri[w[i:i+3]] + 2
                else:
                    relation2grams[idx, j, i] = 1
    np.save('relation2grams', relation2grams)
    np.save('relation2words', relation2words)
    
    
#    words_emb = set()
#    word2idx_emb = {}
#    
#    with open('data/embedding/glove.840B.300d.txt', encoding='utf-8') as fin:
#        for idx, line in enumerate(fin):
#            if idx % 100000 == 0:
#                print(idx,'finished')
#            line = line.strip()
#            if line.split()[0] in words:
#                words_emb.add(line.split()[0])
#                word2idx_emb[line.split()[0]] = ' '.join(line.split()[-300:])
#    
#    
#    print('number of found words:', len(words_emb))
#    words_emb = sorted(words_emb)
#    fout = open('word.dict.emb', 'w', encoding='utf-8')
#    fout.write('<PAD>\n<UNK>\n')
#    for r in words_emb:
#        fout.write(r+'\n')
#    fout.close()
#    
#    fout = open('embedding.300d', 'w', encoding='utf-8')
#    fout.write((str(0.0) + ' ')*300 + '\n' + (str(0.0) + ' ')*300 + '\n')
#    for r in words_emb:
#        fout.write(word2idx_emb[r]+'\n')
#    fout.close()

    words_emb = set()
    with open('word.dict.emb', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if line != '<PAD>' and line != '<UNK>':
                words_emb.add(line)
    
    word2id_emb = {}
    for idx, w in enumerate(words_emb):
        word2id_emb[w] = idx
    relation2words_emb = np.zeros([len(relation), max_len], dtype=np.int32)
    for idx, r in enumerate(relation):
        r = tokenize(r).split()
        for j, w in enumerate(r):
            if w in word2id_emb:
                relation2words_emb[idx, j] = word2id_emb[w] + 2
            else:
                relation2words_emb[idx, j] = 1
    np.save('relation2words_emb', relation2words_emb)
    
    
    
    
    
    
if '__main__' == __name__:
    main()
