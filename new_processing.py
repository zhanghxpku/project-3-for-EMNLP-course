# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:25:22 2019

@author: 38192
"""

import numpy as np
import string

def tokenize(word, num=True):
    ret = ''
    punc = set(string.punctuation)
    for c in word:
        if c not in punc:
            if c.isnumeric() and num:
                if not ret.endswith('0'):
                    ret += '0'
            else:
                ret += c
        else:
            if not ret.endswith(' '):
                ret += ' '
    return ret.strip()

def get_chars(word):
    word = '#' + word + '#'
    chars = [word[i+1] for i in range(len(word)-2)]
    trigrams = [word[i:i+3] for i in range(len(word)-2)]
    return chars, trigrams

def seperate_relation():
    # generate relation set and single relation & CVT datasets
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

        fout1.close()
        fout2.close()
        
def generate_tables():
    # process single-relation
    dataset = ['data/EMNLP.train.single', 'data/EMNLP.dev.single']
    
    words = set()
    relation = set()
    chars = set(['@','#','&'])
    trigrams = set()
    max_len = 0
#    max_len_entity = 0
    
    for idx, ds in enumerate(dataset):
        total_line = 0
        fout = open(ds+'.new', 'w', encoding='utf-8')
        sents = []
        with open(ds, encoding='utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line.startswith('=='):
                    total_line += 1
                    fout.write('\t'.join(sents)+'\n')
                    for item in sents[1:3]:
                        if 'mso:' in item:
                            item = tokenize(item)
                        for word in item.split():
                            words.add(word)
                            if idx == 1:
                                c, t = get_chars(word)
                                chars.update(c)
                                trigrams.update(t)
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
                                relation.add(word)
                                sents.append(word)
                    # entity
                    elif line.startswith('<parameters'):
                        line = line.split('\t')[1]
                        line = line.split()
                        # entity
                        sents.append(line[0])
#                        sents.append(tokenize(line[0]))
#                        max_len_entity = max_len_entity if max_len_entity > len(tokenize(line[0]).split()) else len(tokenize(line[0]).split())
                        # location of entity
                        loc = line[-1][1:-1].split(',')
                        question = sents[1].split()
                        question_new = question[:int(loc[0])]
                        question_new.extend(['@'])
                        question_new.extend(question[int(loc[0]):int(loc[1])+1])
                        question_new.extend(['&'])
                        question_new.extend(question[int(loc[1])+1:])
                        max_len = max_len if max_len > len(question_new) else len(question_new)
                        sents[1] = ' '.join(question_new)
                        
        fout.close()
    
    print('max_len:', max_len)
#    print('longest_entity:', max_len_entity)
    
#    chars = set()
#    trigrams = set()
#    for word in words:
#        word = '#'+word+'#'
#        for idx in range(len(word)-2):
#            trigrams.add(word[idx:idx+3])
#            chars.add(word[idx+1])

    trigrams = sorted(trigrams)
    print('number of trigrams in training set:', len(trigrams))
    fout = open('trigram.dict', 'w', encoding='utf-8')
    fout.write('<PAD>\n<UNK>\n')
    for g in trigrams:
        fout.write(g+'\n')
    fout.close()
    
    chars = sorted(chars)
    print('number of chars in training set:', len(chars))
    fout = open('char.dict', 'w', encoding='utf-8')
    fout.write('<PAD>\n<UNK>\n')
    for c in chars:
        fout.write(c+'\n')
    fout.close()

    print('number of relations all sets:', len(relation))
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
    
    return words, chars, trigrams, relation    

def build_maps(words, chars, trigrams, relation):
    word_num = len(words)
    max_char = 0
    for word in words:
        max_char = max_char if max_char > len(word) else len(word)
    
    print('longest words:', max_char)

    char2id = {}
    for idx, c in enumerate(chars):
        char2id[c] = idx + 2
    
    trigram2id = {}
    for idx, g in enumerate(trigrams):
        trigram2id[g] = idx + 2
        
    word2id = {}
    for idx, w in enumerate(words):
        word2id[w] = idx + 2
    
    word2char = np.zeros(shape=[word_num + 2, max_char], dtype=np.int32)
    word2gram = np.zeros(shape=[word_num + 2, max_char], dtype=np.int32)
    for idx, word in enumerate(words):
        word = '#' + word + '#'
        for i in range(len(word)-2):
            if word[i+1] in chars:
                word2char[idx+2, i] = char2id[word[i+1]]
            else:
                word2char[idx+2, i] = 1
            if word[i:i+3] in trigrams:
                word2gram[idx+2, i] = trigram2id[word[i:i+3]]
            else:
                word2gram[idx+2, i] = 1
    
    np.save('word2char', word2char)
    np.save('word2gram', word2gram)
    
    relation_num = len(relation)
    max_word = 0
    for r in relation:
        r = r.split('mso:')[-1]
        r = tokenize(r).split()
        max_word = max_word if max_word > len(r) else len(r)
    
    print('longest relations:', max_word)
    
    relation2word = np.zeros(shape=[relation_num, max_word], dtype=np.int32)
    relation2char = np.zeros(shape=[relation_num, max_word, max_char], dtype=np.int32)
    relation2gram = np.zeros(shape=[relation_num, max_word, max_char], dtype=np.int32)
    
    for idx_r, r in enumerate(relation):
        r = r.split('mso:')[-1]
        r = tokenize(r).split()
        for idx, word in enumerate(r):
            if word in words:
                relation2word[idx_r, idx] = word2id[word]
            else:
                relation2word[idx_r, idx] = 1
            word = '#' + word + '#'
            for i in range(len(word)-2):
                if word[i+1] in chars:
                    relation2char[idx_r, idx, i] = char2id[word[i+1]]
                else:
                    relation2char[idx_r, idx, i] = 1
                if word[i:i+3] in trigrams:
                    relation2gram[idx_r, idx, i] = trigram2id[word[i:i+3]]
                else:
                    relation2gram[idx_r, idx, i] = 1
    
    np.save('relation2char', relation2char)
    np.save('relation2word', relation2word)
    np.save('relation2gram', relation2gram)
    
def build_pretrained(words, chars, trigrams, relation, read_glove=True):
    words_emb = set()
    word2idx_emb = {}
    
    if read_glove:
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
        fout = open('word_emb.dict', 'w', encoding='utf-8')
        fout.write('<PAD>\n<UNK>\n')
        for r in words_emb:
            fout.write(r+'\n')
        fout.close()
        
        fout = open('embedding.300d', 'w', encoding='utf-8')
        fout.write((str(0.0) + ' ')*300 + '\n' + (str(0.0) + ' ')*300 + '\n')
        for r in words_emb:
            fout.write(word2idx_emb[r]+'\n')
        fout.close()
    else:
        words_emb = set()
        with open('word_emb.dict', encoding='utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line != '<PAD>' and line != '<UNK>':
                    words_emb.add(line)

    max_char = 0
    for word in words:
        max_char = max_char if max_char > len(word) else len(word)
    print('longest words:', max_char)
    
    char2id = {}
    for idx, c in enumerate(chars):
        char2id[c] = idx + 2
    
    trigram2id = {}
    for idx, g in enumerate(trigrams):
        trigram2id[g] = idx + 2
        
    word2id = {}
    for idx, w in enumerate(words_emb):
        word2id[w] = idx + 2
    
    word2char = np.zeros(shape=[len(words_emb) + 2, max_char], dtype=np.int32)
    word2gram = np.zeros(shape=[len(words_emb) + 2, max_char], dtype=np.int32)
    for idx, word in enumerate(words_emb):
        word = '#' + word + '#'
        for i in range(len(word)-2):
            if word[i+1] in chars:
                word2char[idx+2, i] = char2id[word[i+1]]
            else:
                word2char[idx+2, i] = 1
            if word[i:i+3] in trigrams:
                word2gram[idx+2, i] = trigram2id[word[i:i+3]]
            else:
                word2gram[idx+2, i] = 1
    
    np.save('word2char_emb', word2char)
    np.save('word2gram_emb', word2gram)
    
    relation_num = len(relation)
    max_word = 0
    for r in relation:
        r = r.split('mso:')[-1]
        r = tokenize(r).split()
        max_word = max_word if max_word > len(r) else len(r)
    
    print('longest relations:', max_word)
    
    relation2word = np.zeros(shape=[relation_num, max_word], dtype=np.int32)
    for idx_r, r in enumerate(relation):
        r = r.split('mso:')[-1]
        r = tokenize(r).split()
        for idx, word in enumerate(r):
            if word in words_emb:
                relation2word[idx_r, idx] = word2id[word]
            else:
                relation2word[idx_r, idx] = 1
    
    np.save('relation2word_emb', relation2word)
    

def main():
    # seperate into single relation and CVT
    seperate_relation()
    # process single relation
    words, chars, trigrams, relation = generate_tables()
    build_maps(words, chars, trigrams, relation)
    build_pretrained(words, chars, trigrams, relation, read_glove=True)

    
if '__main__' == __name__:
    main()