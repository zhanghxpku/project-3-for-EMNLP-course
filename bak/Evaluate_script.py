#_*_coding=utf-8_*_
# This python script use python3.5
import sys
def instance2dict(instance):
	# convert the instance into a dict
	tmp_dict = {}
	tmp_dict['question'] = instance[0].split('\t')[1]
	tmp_dict['logical_form'] = instance[1].split('\t')[1]
	tmp_dict['parameters'] = instance[2].split('\t')[1]
	tmp_dict['question_type'] = instance[3].split('\t')[1]
	return tmp_dict

def fetch_data(file):
	# fetch data from the original file
	lines = file.readlines()
	instance_list = []
	tmp_instance = []
	for line in lines:
		if line.strip()=="==================================================":
			instance_list.append(instance2dict(tmp_instance))
			tmp_instance = []
		else:
			tmp_instance.append(line)
	return instance_list

def main():
	#a script to calculate the EM score, the input format is: python Evaluate_script.py PREDICTED_FILENAME GOLD_FILENAME
	print("The predicted file is %s" %(sys.argv[1]))
	print("The gold file is %s" %(sys.argv[2]))
	predicted_file = open(sys.argv[1], 'r', encoding='utf-8')
	gold_file = open(sys.argv[2], 'r', encoding='utf-8')
	predicted_data = fetch_data(predicted_file)
	gold_data = fetch_data(gold_file)
	if len(predicted_data) != len(gold_data):
		print("ERROR: The number of prediction is different with the number of gold data!!!!")
		exit(1)
	total_count = 0.0
	right_count = 0.0
	for i in range(len(gold_data)):
		total_count += 1
		if predicted_data[i]['logical_form']==gold_data[i]['logical_form']:
			right_count += 1
	print("The EM score is %f" %(float(right_count)/total_count))
if __name__=="__main__":
	main()
