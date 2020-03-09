import re
import string
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
import random
import pprint
pp = pprint.PrettyPrinter(indent = 4)

'''
#from internet
def dict_path(path,my_dict):
    dictArray = []

    for k,v in my_dict.items():
        if isinstance(v,dict):
            dict_path(path+" "+k,v)
        else:
            print(path +" " + k)
            noNewLines = path +" " + k
            noNewLines = re.split(r"\n", noNewLines)

            words = noNewLines
            words = re.split(r" ", words)
            words = words[1:]
            
            for index in range(0, len(words), 2):
                currD = {}
                currD[words[index]] = words[index+1]
                dictArray.append(currD)
                # print(currD)

            # print(words)
            # print(path+" "+k,"=>",v)
    print(dictArray)
#dict_path("",my_dict)
'''

def id3(s, attributes, classes):

    if len(s) == 0:
        return

    current_classes = return_classes(s)

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
    temp = [ [ 0 for y in range( len(classes)+1 ) ] for x in range( len(variables) ) ]
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



def get_entropy(count, total, count2):
    if count/total <= 0:
        return 0

    return -(count/total*math.log2(count/total)+count2/total*math.log2(count2/total))


def classify_me(example, dictionary, index_labels):

    classification = ""

    for key in dictionary.keys():
        print('key is:', key)
        currKey = key

        index = index_labels[key]
        currentVariable = example[index]
        print('currentVariable:', currentVariable)

        for key2 in dictionary[currKey].keys():

            print('key2 is:', key2)

            if currentVariable == key2:
                print('is instance', isinstance(dictionary[currKey][key2],dict))
                if isinstance(dictionary[currKey][key2],dict):
                    classification = classify_me(example, dictionary[currKey][key2], index_labels)
                else:
                    print('inside else. here is currkey:', dictionary[currKey][key2])
                    return dictionary[currKey][key2]

    return classification



#read in contents
file = open("contactData.txt", "r")
contents = file.read()
info = re.split(r"\n", contents)

#create dictionary with information
fileDictionary = {"s": []}
for index in range(0, len(info)):
    if index == 0:
        # class values
        fileDictionary["classes"] = {"num": int(info[index]), "values": info[1].split(",")}
        
    if index == 2:
        fileDictionary["attributes"] = {"num":int(info[index])}
        for attributeIndex in range(index+1, index+1+int(info[index])):
            attribute = info[attributeIndex].split(",")[0]
            numValues = info[attributeIndex].split(",")[1]
            values = info[attributeIndex].split(",")[2:]
            fileDictionary["attributes"][attribute] = {"index": attributeIndex-3, "num":numValues, "values":values}   
    
    if index == 7:
        fileDictionary["total"] = int(info[index])
    
    if index > 7:
        fileDictionary["s"].append(info[index])

yesCount = 0
noCount = 0
totalNumData = 0

#step 1, get total entropy
for line in info[8:]:
    lineArr = re.split(r",", line)
    totalNumData = totalNumData + 1

    if(lineArr[len(lineArr)-1])=='Yes':
        yesCount += 1
    else:
        noCount += 1


entropy = get_entropy(yesCount, totalNumData, noCount)
sArray = fileDictionary['s']



random.shuffle(sArray)
amountOfTraining = round(len(sArray)*.8)
training = sArray[0:amountOfTraining]
test = sArray[amountOfTraining:]
final = id3(training, fileDictionary["attributes"], fileDictionary["classes"])
index_labels = {'age': 0, 'prescription': 1, 'astigmatism': 2, 'tear-rate':3}


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
print("Incorect classifications", incorrect_classification)

#pp.pprint(final)
