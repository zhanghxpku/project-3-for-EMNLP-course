#!/usr/bin/env python
# -*- coding: utf-8 -*-

test_data = '../data/EMNLP.test'
infer_data = 'infer.output'
output_data = 'infer.output.new'

infer_map = []
entity_map = []

with open(test_data, encoding='utf-8') as fin:
    for line in fin:
        if line.startswith('<parameter'):
            line = line.strip().split('\t')
            line = line[-1].split()
            if '|||' in line:
                entity = [line[0],line[4]]
            else:
                entity = [line[0]]
            entity_map.append(entity)

with open(infer_data, encoding='utf-8') as fin:
    for line in fin:
        line = line.strip().split('\t')
        idx = int(line[0])
        relation = line[1]
        typ = int(line[2])
        order = int(line[3])
        entity = entity_map[idx-1]
        if typ == 1 and order == 1:
            entity = [entity[1], entity[0]]
        if typ == 0:
            argument = '( lambda ?x ( ' + relation + ' ' + entity[0] + ' ?x ) )'
        else:
            relation = relation.split()
            argument = '( lambda ?x exist ?y ( and ( ' + relation[0] + ' ' + entity[0] + ' ?y ) ( ' + relation[1] + ' ?y ' + entity[1] + ' ) ( ' + relation[2] + ' ?y ?x ) ) )'
        infer_map.append(argument)

fout = open(output_data, 'w', encoding='utf-8')
with open(test_data, encoding='utf-8') as fin:
    for idx, line in enumerate(fin):
        if line.startswith('<logical'):
            line = line.strip() + ' ' + infer_map[int(idx/5)] + '\n'
        else:
            line = line
        fout.write(line)
            
fout.close()
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            