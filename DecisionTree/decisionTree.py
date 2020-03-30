import re
import string
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
import random
import pprint
import pydot
pp = pprint.PrettyPrinter(indent = 4)


#source: https://stackoverflow.com/questions/13688410/dictionary-object-to-decision-tree-in-pydot
def draw(parent_name, child_name):
    edge = pydot.Edge(parent_name, child_name)
    graph.add_edge(edge)

def visit(node, parent=None):
    for k,v in node.items():
        if isinstance(v, dict):
            # We start with the root node whose parent is None
            # we don't want to graph the None node
            if parent:
                draw(parent, k)
            visit(v, k)
        else:
            draw(parent, k)
            # drawing the label using a distinct name
            draw(k, k+'_'+v)

#source https://stackoverflow.com/questions/34836777/print-complete-key-path-for-all-the-values-of-a-python-nested-dictionary
def dict_path(path,my_dict):

    for k,v in my_dict.items():
        if isinstance(v,dict):
            dict_path(path+" "+k,v)
        else:
            line = path+ " " +k
            line = re.split(r" ", line)
            line = line[1:]
            fullLine = "if "

            for index in range(0, len(line)-1, 2):
                if index == len(line)-2:
                    fullLine = fullLine + line[index] + " is " + line[index+1]

                else:
                    fullLine = fullLine + line[index] + " is " + line[index+1] + " and "
            
            fullLine = fullLine + ", then recommendation is: " + v
            print(fullLine)
            

def id3(s, attributes, classes):

    current_classes = return_classes(s)

    if len(s) == 0:
        return

    #if all examples in s are of the same class
    if len(current_classes.keys()) == 1:
        for key in current_classes.keys():
            return key

    #else if no attributes left
    elif attributes['num'] == 0:
        return get_most_common_class(current_classes)

    #else choose the attribute that maximizes the gain 
    else:
        attribute_with_most_gain = get_most_gain(s, attributes, classes)
        currentDict = {}
        for instance in s:
            instanceIndexed = re.split(r",", instance)

            for value in attributes[attribute_with_most_gain]['values']:
                
                if value not in currentDict:
                    currentDict[value] = []
                if instanceIndexed[attributes[attribute_with_most_gain]['index']] == value:
                    currentDict[value].append(instance)
        

        del attributes[attribute_with_most_gain]
        attributes["num"] = attributes['num'] - 1
        
        for key, value in currentDict.items():
            attributes2 = copy.deepcopy(attributes)
            currentDict[key] = id3(value, attributes2, classes)
        

        finalDict = {attribute_with_most_gain: currentDict}
        return finalDict


def get_most_gain(s, attributes, classes):
    
    keys = attributes.keys()
    gains = {}
    for key in keys:
        if key != 'num':
            gains[key] = get_gain(attributes[key]['values'], s, attributes[key]['index'], classes)
    
    #return the maximum of the gains
    return max(gains, key=gains.get)



def get_gain(variables, instances, index, classes):

    temp = [ [ 0 for y in range( len(classes['values']) ) ] for x in range( len(variables) ) ]

    gain = []
    for instance in instances:
        instance = re.split(r",", instance)

        for j, variable in enumerate(variables):
            #index is the position of the attribute
            if instance[index] == variable:

                for k, class_a in enumerate(classes['values']):
                    if instance[len(instance)-1] == class_a:
                        if temp[j][k] == 0:
                            temp[j][k] = 1

                        else:
                            temp[j][k] = temp[j][k] + 1


    for i, line in enumerate(temp):
        summation = 0
        total = 0

        for val in line:

            if val > 0 and sum(line) > 0:
                summation = summation + val/sum(line)*math.log2(val/sum(line))
                total = total + val
            else:
                summation = summation + 0
                total = total + val

        currEntropy = summation

        if len(instances)*summation != 0:
            entropyForFormula = total/len(instances)*summation
        else:
            entropyForFormula = 0

        gain.append(entropyForFormula)

    finalGain = entropy + sum(gain)
    return finalGain
    

def get_most_common_class(current_classes):
    return max(current_classes, key=current_classes.get)


def return_classes(s):
    current_classes = {}

    for example in s:
        example = re.split(r",", example)
        if example[len(example)-1] in current_classes:
            current_classes[example[len(example)-1]] = current_classes[example[len(example)-1]] + 1
        else:
            current_classes[example[len(example)-1]] = 1

    return current_classes



def get_entropy(s):

    countingDict = {}
    total = 0
    for instance in s:

        instance = re.split(r",", instance)
        total = total + 1

        if instance[len(instance)-1] in countingDict:
            countingDict[instance[len(instance)-1]] = countingDict[instance[len(instance)-1]] + 1
        else:
           countingDict[instance[len(instance)-1]] = 1

    entropyList = []
    for val in countingDict.values():
        value = val/total*math.log2(val/total)
        entropyList.append(value)

    finalEntropy = -sum(entropyList)
    return finalEntropy


def classify_me(example, dictionary, index_labels):

    classification = ""

    for key in dictionary.keys():
        currKey = key
        index = index_labels[key]
        currentVariable = example[index]

        for key2 in dictionary[currKey].keys():

            if currentVariable == key2:
                if isinstance(dictionary[currKey][key2],dict):
                    classification = classify_me(example, dictionary[currKey][key2], index_labels)
                else:
                    return dictionary[currKey][key2]

    return classification

def get_index_labels(attributes):

    index_labels = {}
    for key in attributes.keys():
        if key != 'num':
            index_labels[key] = attributes[key]['index']

    return index_labels

def is_an_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

#read in contents
file = open("carData.txt", "r")
contents = file.read()
info = re.split(r"\n", contents)

#create dictionary with information
fileDictionary = {"s": []}
count = 0
for index in range(0, len(info)):

    if is_an_int(info[index]) and count == 0:
        # class values
        fileDictionary["classes"] = {"num": int(info[index]), "values": info[1].split(",")}
        count = count + 1
        
    elif is_an_int(info[index]) and count == 1:
        fileDictionary["attributes"] = {"num":int(info[index])}
        count = count + 1
       
        for attributeIndex in range(index+1, index+1+int(info[index])):
            attribute = info[attributeIndex].split(",")[0]
            numValues = info[attributeIndex].split(",")[1]
            values = info[attributeIndex].split(",")[2:]
            fileDictionary["attributes"][attribute] = {"index": attributeIndex-3, "num":numValues, "values":values}
    
    elif is_an_int(info[index]) and count == 2:
        fileDictionary["total"] = int(info[index])
        count = count + 1
    
    elif count == 3:
        fileDictionary["s"].append(info[index])


entropy = get_entropy(fileDictionary['s'])
sArray = fileDictionary['s']
copyOfInfo = copy.deepcopy(fileDictionary)

random.shuffle(sArray)
amountOfTraining = round(len(sArray)*.8)
training = sArray[0:amountOfTraining]
test = sArray[amountOfTraining:]

final = id3(training, fileDictionary["attributes"], fileDictionary["classes"])
index_labels = get_index_labels(copyOfInfo["attributes"])

#pp.pprint(final)
#dict_path("", final)

correct_classification = 0
incorrect_classification = 0

for example in test:
    example = re.split(r",", example)
    classification = classify_me(example, final, index_labels)
    if classification == example[len(example)-1]:
        correct_classification = correct_classification + 1
    else:
        incorrect_classification = incorrect_classification + 1

print("Correct classifications", correct_classification)
print("Incorrect classifications", incorrect_classification)

'''
#graphing with pydot
graph = pydot.Dot(graph_type='graph')
visit(final)
graph.write_png('fishing_graph2.png')
'''