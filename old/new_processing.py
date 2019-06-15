# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:25:22 2019

@author: 38192
"""

import numpy as np
#import string

def tokenize(word, num=False, pun=True):
    ret = ''
    punc = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~–’⁄「」')
    for c in word:
        if not pun or c not in punc:
            if c.isnumeric() and num:
                if not ret.endswith('0'):
                    ret += '0'
            else:
                ret += c
        else:
            if not ret.endswith(' '):
                ret += ' '
    temp = []
    for w in ret.split():
        has_num = False
        for c in w:
            if c.isnumeric() and num:
                has_num = True
                break
        if has_num:
            temp.append('0')
        else:
            temp.append(w)
    return ' '.join(temp).strip()

def get_chars(word, pad=False):
    if not word.startswith('#'):
        word = '#' + word + '#'
    if not pad:
        chars = [word[i+1] for i in range(len(word)-2)]
        trigrams = [word[i:i+3] for i in range(len(word)-2)]
    else:
        chars = [word[i+1].replace('#', '<PAD>') for i in range(len(word)-2)]
        trigrams = [word[i:i+3] for i in range(len(word)-2)]
        for t in range(len(word)-2):
            trigrams[t] = trigrams[t] if trigrams[t][1] != '#' else '<PAD>'
    return chars, trigrams

def seperate_relation():
    # generate relation set and single relation & CVT datasets
    train_dataset = ['data/EMNLP.train']
    dataset = ['data/EMNLP.dev', 'data/EMNLP.test']
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
    dataset = ['data/EMNLP.train', 'data/EMNLP.dev']
    
    words = set()
    words_all = set()
    relation = set()
    relation_comb = set()
    relation_single = set()
    chars = set(['#','&','#'])
    trigrams = set()
    entity = set()
    entity_comb = set()
    max_len = 0
    max_char = 0
    char_len = {}
    
    for idx, ds in enumerate(dataset):
        total_line = 0
        fout = open(ds+'.new', 'w', encoding='utf-8')
        sents = []
        with open(ds, encoding='utf-8') as fin:
            question_type = ''
            ent_pair = []
            for line in fin:
                line = line.strip()
                if line.startswith('=='):
                    total_line += 1
                    sents.append(tokenize(sents[1],num=True).replace('bbbbb','@').replace('eeeee', '&'))
                    sents[1] = tokenize(sents[1]).replace('bbbbb','@').replace('eeeee', '&')
                    max_len = max_len if max_len > len(sents[1].split()) else len(sents[1].split())
                    assert len(sents[1].split()) == len(sents[-1].split()), (len(sents[1].split()),len(sents[-1].split()))
                    sents_char = '#' + '#'.join(sents[1].split()) + '#'
                    c, t = get_chars(sents_char, pad=True)
                    sents[1] = ' '.join(c)
                    sents.append(' '.join(t))
                    if len(sents_char) in char_len:
                        char_len[len(sents_char)] += 1
                    else:
                        char_len[len(sents_char)] = 1
                    max_char = max_char if max_char > len(sents_char) else len(sents_char)
                    if sents[3] == ' '.join(ent_pair):
                        sents.append('0')
                    elif sents[3] != ' '.join(ent_pair):
                        sents.append('1')
                    ent_pair = []
                    sents.append('0' if question_type=='single' else '1')
                    perm = [0,1,5,4,2,3,7,6]
                    sents_new = [sents[i] for i in perm]
                    mask = np.zeros([4])
                    mask_char = np.zeros([4])
                    count = 0
                    for i, c in enumerate(sents_new[1].split()):
                        if c == '@':
                            mask_char[count*2] = i
                        elif c == '&':
                            mask_char[count*2+1] = i
                            count += 1
                    count = 0
                    for i, c in enumerate(sents_new[3].split()):
                        if c == '@':
                            mask[count*2] = i
                        elif c == '&':
                            mask[count*2+1] = i
                            count += 1
                    
                    entity_2 = np.zeros([len(sents_new[3])])
                    for i in range(int(mask[0]), int(mask[1])):
                        entity_2[i] = 1
                    sents_new.append(' '.join([str(int(i)) for i in entity_2]))
                    entity_2 = np.zeros([len(sents_new[3])])
                    for i in range(int(mask[2]), int(mask[3])):
                        entity_2[i] = 1
                    sents_new.append(' '.join([str(int(i)) for i in entity_2]))

                    entity_1 = np.zeros([len(sents_new[1])])
                    for i in range(int(mask_char[0]), int(mask_char[1])):
                        entity_1[i] = 1
                    sents_new.append(' '.join([str(int(i)) for i in entity_1]))
                    entity_1 = np.zeros([len(sents_new[1])])
                    for i in range(int(mask_char[2]), int(mask_char[3])):
                        entity_1[i] = 1
                    sents_new.append(' '.join([str(int(i)) for i in entity_1]))
                    
#                    sents_new.append(' '.join([str(int(i)) for i in mask]))
#                    sents_new.append(' '.join([str(int(i)) for i in mask_char]))

                    fout.write('\t'.join(sents_new)+'\n'.replace('  ', ' '))
                    if idx == 0:
                        chars.update(sents_new[1].split())
                        trigrams.update(sents_new[2].split())
                        words.update(sents_new[3].split())
                    words_all.update(sents_new[3].split())
                    sents = []
                else:
                    # question
                    if line.startswith('<question id'):
                        sents.append(line.split('\t')[0].split('=')[-1][:-1])
                        sents.append(line.split('\t')[1])
                    #relation
                    elif line.startswith('<logical'):
                        line = line.split('\t')[1]
                        line = line.split()
                        if len(line) > 18:
                            question_type = 'cvt'
                        else:
                            question_type = 'single'
                        if question_type == 'single':
                            word = line[4]
                            relation.add(word)
                            sents.append(word)
                            relation_single.add(word)
                        else:
                            word = [line[8], line[13], line[18]]
                            relation_comb.add(' '.join(word))
                            ent_pair = [line[9], line[15]]
                            for w in word:
                                relation.add(w)
                            sents.append(' '.join(word))
                    # entity
                    elif line.startswith('<parameters'):
                        line = line.split('\t')[1]
                        line = line.split()
                        # entity
                        sents.append(line[0])
                        entity.add(line[0])
                        # location of entity
                        loc = line[2][1:-1].split(',')
                        question = sents[1].split()
                        question_new = question[:int(loc[0])]
                        question_new.extend(['bbbbb'])
                        question_new.extend(question[int(loc[0]):int(loc[1])+1])
                        question_new.extend(['eeeee'])
                        question_new.extend(question[int(loc[1])+1:])
                        max_len = max_len if max_len > len(question_new) else len(question_new)
                        sents[1] = ' '.join(question_new)
                        if question_type == 'cvt':
                            # second entity
                            sents[-1] = sents[-1] + ' '+line[4]
                            entity_comb.add(sents[-1])
                            entity.add(line[4])
                            # location of entity
                            loc = line[-1][1:-1].split(',')
                            question = sents[1].split()
                            question_new = question[:int(loc[0])+2]
                            question_new.extend(['bbbbb'])
                            question_new.extend(question[int(loc[0])+2:int(loc[1])+3])
                            question_new.extend(['eeeee'])
                            question_new.extend(question[int(loc[1])+3:])
                            sents[1] = ' '.join(question_new)
                        
        fout.close()
    
    print('max_len:', max_len)
    print('max_char:', max_char)
#    print('longest_entity:', max_len_entity)
    
    max_len = 0
    max_char = 0

    for r in relation:
        r = r.split('mso:')[-1]
        r_word = tokenize(r,num=True).split()
        words.update(r_word)
        words_all.update(r_word)
        r = tokenize(r).split()
        max_len = max_len if max_len > len(r) else len(r)
        sents_char = '#' + '#'.join(r) + '#'
        c, t = get_chars(sents_char, pad=True)
        chars.update(c)
        trigrams.update(t)
        max_char = max_char if max_char > len(sents_char) else len(sents_char)
    
    print('max_len:', max_len)
    print('max_char:', max_char)
    
    trigrams.remove('<PAD>')
    trigrams = sorted(trigrams)
    print('number of trigrams in training set:', len(trigrams))
    fout = open('trigram.dict', 'w', encoding='utf-8')
    fout.write('<PAD>\n<UNK>\n')
    for g in trigrams:
        fout.write(g+'\n')
    fout.close()
    
    chars.remove('<PAD>')
    chars = sorted(chars)
    print('number of chars in training set:', len(chars))
    fout = open('char.dict', 'w', encoding='utf-8')
    fout.write('<PAD>\n<UNK>\n')
    for c in chars:
        fout.write(c+'\n')
    fout.close()

    print('number of relations all sets:', len(relation))
    relation = sorted(relation)
    fout = open('relation.dict', 'w', encoding='utf-8')
#    fout.write('<PAD>\n<UNK>\n')
    for r in relation:
        fout.write(r+'\n')
    fout.close()
    
    print('number of relation combinations in all sets:', len(relation_comb))
    relation_comb = sorted(relation_comb)
    fout = open('relation_comb.dict', 'w', encoding='utf-8')
#    fout.write('<PAD>\n<UNK>\n')
    for r in relation_comb:
        fout.write(r+'\n')
    fout.close()
    
    print('number of relation combinations in single questions:', len(relation_single))
    relation_single = sorted(relation_single)
    fout = open('relation_single.dict', 'w', encoding='utf-8')
#    fout.write('<PAD>\n<UNK>\n')
    for r in relation_single:
        fout.write(r+'\n')
    fout.close()
    
    print('number of entities in all sets:', len(entity))
    entity = sorted(entity)
    fout = open('entity.dict', 'w', encoding='utf-8')
#    fout.write('<PAD>\n<UNK>\n')
    for e in entity:
        fout.write(e+'\n')
    fout.close()
    
    print('number of entity combination in all sets:', len(entity_comb))
    entity_comb = sorted(entity_comb)
    fout = open('entity_comb.dict', 'w', encoding='utf-8')
#    fout.write('<PAD>\n<UNK>\n')
    for e in entity_comb:
        fout.write(e+'\n')
    fout.close()
    
    print('number of words in all sets:', len(words_all))
    words_all = sorted(words_all)
    fout = open('word_all.dict', 'w', encoding='utf-8')
    fout.write('<PAD>\n<UNK>\n')
    for r in words_all:
        fout.write(r+'\n')
    fout.close()
    
    print('number of words in training sets:', len(words))
    words = sorted(words)
    fout = open('word.dict', 'w', encoding='utf-8')
    fout.write('<PAD>\n<UNK>\n')
    for r in words:
        fout.write(r+'\n')
    fout.close()
    
    return words, chars, trigrams, relation, relation_comb, relation_single, max_len, max_char, words_all, char_len

def build_maps(words, chars, trigrams, relation, relation_comb, relation_single, max_len, max_char, words_all):
    char2id = {}
    for idx, c in enumerate(chars):
        char2id[c] = idx + 2
    
    trigram2id = {}
    for idx, g in enumerate(trigrams):
        trigram2id[g] = idx + 2
        
    word2id = {}
    for idx, w in enumerate(words):
        word2id[w] = idx + 2
    
    relation_num = len(relation)
    
    relation2word = np.zeros(shape=[relation_num, max_len], dtype=np.int32)
    relation2char = np.zeros(shape=[relation_num, max_char], dtype=np.int32)
    relation2gram = np.zeros(shape=[relation_num, max_char], dtype=np.int32)
    
    for idx_r, r in enumerate(relation):
        r = r.split('mso:')[-1]
        r_word = tokenize(r,num=True).split()
        for idx, word in enumerate(r_word):
            if word in words:
                relation2word[idx_r, idx] = word2id[word]
            else:
                relation2word[idx_r, idx] = 1
        r = tokenize(r).split()
        sents_char = '#' + '#'.join(r) + '#'
        c, t = get_chars(sents_char, pad=True)
        for idx, char in enumerate(c):
            if char in chars:
                relation2char[idx_r, idx] = char2id[char]
            else:
                relation2char[idx_r, idx] = 1
        for idx, trigram in enumerate(t):
            if trigram in trigrams:
                relation2gram[idx_r, idx] = trigram2id[trigram]
            else:
                relation2gram[idx_r, idx] = 1
    
    np.save('relation2char', relation2char)
    np.save('relation2word', relation2word)
    np.save('relation2gram', relation2gram)
    
    word2id = {}
    for idx, w in enumerate(words_all):
        word2id[w] = idx + 2
    
    relation2word_all = np.zeros(shape=[relation_num, max_len], dtype=np.int32)
    for idx_r, r in enumerate(relation):
        r = r.split('mso:')[-1]
        r = tokenize(r,num=True).split()
        for idx, word in enumerate(r):
            if word in words_all:
                relation2word_all[idx_r, idx] = word2id[word]
            else:
                relation2word_all[idx_r, idx] = 1
    
    np.save('relation2word_all', relation2word_all)
    
    relation2id = {}
    for idx, r in enumerate(relation):
        relation2id[r] = idx
    
    comb2relation = np.zeros(shape=[len(relation_single)+len(relation_comb), 3], dtype=np.int32)
    for idx, c in enumerate(relation_single):
        comb2relation[idx,0] = relation2id[c]
        comb2relation[idx,1] = relation2id[c]
        comb2relation[idx,2] = relation2id[c]
    for idx, c in enumerate(relation_comb):
        for j, relations in enumerate(c.split()):
            comb2relation[len(relation_single)+idx,j] = relation2id[relations]
    np.save('comb2relation', comb2relation)
    fout = open('comb.dict', 'w', encoding='utf-8')
    for r in relation_single:
        fout.write(r+'\n')
    for r in relation_comb:
        fout.write(r+'\n')
    fout.close()
    
    max_char = 0
    for word in words_all:
        max_char = max_char if max_char > len(word) else len(word)
    print('longest word all:', max_char)
    word2char = np.zeros([len(words_all)+2, max_char], dtype=np.int32)
    word2gram = np.zeros([len(words_all)+2, max_char], dtype=np.int32)
    for idx, word in enumerate(words_all):
        word = '#' + word + '#'
        for i in range(len(word)-2):
            if word[i+1] in char2id:
                word2char[idx+2, i] = char2id[word[i+1]]
            else:
                word2char[idx+2, i] = 1
            if word[i:i+3] in trigram2id:
                word2gram[idx+2, i] = trigram2id[word[i:i+3]]
            else:
                word2gram[idx+2, i] = 1
            
    np.save('word2char', word2char)
    np.save('word2gram', word2gram)
    
def build_pretrained(words, chars, trigrams, relation, read_glove=True):
#    words = set([tokenize(word,num=True,pun=False) for word in words])
#    words = sorted(words)
    print(len(words))
    words_emb = set()
    word2idx_emb = {}
    
    if read_glove:
        with open('data/embedding/glove.840B.300d.txt', encoding='utf-8') as fin:
            for idx, line in enumerate(fin):
                if idx % 100000 == 0:
                    print(idx,'finished')
                line = line.strip().split()
                if line[0] in words:
                    words_emb.add(line[0])
                    word2idx_emb[line[0]] = ' '.join(line[-300:])
        
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
        with open('word_emb.dict', encoding='utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line != '<PAD>' and line != '<UNK>':
                    words_emb.add(line)
        words_emb = sorted(words_emb)
        
    word2id = {}
    for idx, w in enumerate(words_emb):
        word2id[w] = idx + 2
    
    relation_num = len(relation)
    max_word = 0
    for r in relation:
        r = r.split('mso:')[-1]
        r = tokenize(r,num=True).split()
        max_word = max_word if max_word > len(r) else len(r)
    
    print('longest relations:', max_word)
    
    relation2word = np.zeros(shape=[relation_num, max_word], dtype=np.int32)
    for idx_r, r in enumerate(relation):
        r = r.split('mso:')[-1]
        r = tokenize(r,num=True).split()
        for idx, word in enumerate(r):
            if word in words_emb:
                relation2word[idx_r, idx] = word2id[word]
            else:
                relation2word[idx_r, idx] = 1
    
    np.save('relation2word_emb', relation2word)
    
    char2id = {}
    for idx, c in enumerate(chars):
        char2id[c] = idx + 2
    
    trigram2id = {}
    for idx, g in enumerate(trigrams):
        trigram2id[g] = idx + 2
    
    max_char = 0
    for word in words_emb:
        max_char = max_char if max_char > len(word) else len(word)
    print('longest word emb:', max_char)
    word2char_emb = np.zeros([len(words_emb)+2, max_char], dtype=np.int32)
    word2gram_emb = np.zeros([len(words_emb)+2, max_char], dtype=np.int32)
    for idx, word in enumerate(words_emb):
        word = '#' + word + '#'
        for i in range(len(word)-2):
            if word[i+1] in char2id:
                word2char_emb[idx+2, i] = char2id[word[i+1]]
            else:
                word2char_emb[idx+2, i] = 1
            if word[i:i+3] in trigram2id:
                word2gram_emb[idx+2, i] = trigram2id[word[i:i+3]]
            else:
                word2gram_emb[idx+2, i] = 1

    np.save('word2char_emb', word2char_emb)
    np.save('word2gram_emb', word2gram_emb)
    
    return words_emb

    

#def main():
## seperate into single relation and CVT
#seperate_relation()
words, chars, trigrams, relation, comb, relation_single, max_len, max_char, words_all, char_len = generate_tables()
build_maps(words, chars, trigrams, relation, comb, relation_single, max_len, max_char, words_all)
words_emb = build_pretrained(words_all, chars, trigrams, relation, read_glove=False)


#
#    
#if '__main__' == __name__:
#    main()