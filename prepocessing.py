#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os



def main():
    """main"""
    


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
    
    
    words_emb = set()
    word2idx_emb = {}
    
    with open('data/embedding/glove.840B.300d.txt', encoding='utf-8') as fin:
        for idx, line in enumerate(fin):
            if idx % 100000 == 0:
                print(idx,'finished')
            line = line.strip()
            if line.split()[0] in words:
                words_emb.add(line.split()[0])
                word2idx_emb[line.split()[0]] = ' '.join(line.split()[-300:])
    
    
    print('number of found words:', len(words_emb))
    words_emb = sorted(words_emb)
    fout = open('word.dict.emb', 'w', encoding='utf-8')
    fout.write('<PAD>\n<UNK>\n')
    for r in words_emb:
        fout.write(r+'\n')
    fout.close()
    
    fout = open('embedding.300d', 'w', encoding='utf-8')
    fout.write((str(0.0) + ' ')*300 + '\n' + (str(0.0) + ' ')*300 + '\n')
    for r in words_emb:
        fout.write(word2idx_emb[r]+'\n')
    fout.close()

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
