import re
import string
import matplotlib.pyplot as plt
import numpy as np
import math


def id3(s, attributes, classes, returnableDict):

    if len(s) == 0:
        return

    current_classes = return_classes(s)

    #if all examples in s are of the same class
    if len(current_classes.keys()) <= 1:
        return current_classes.keys()

    #else if no attributes left
    if attributes['num'] == 0:
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
                elif instanceIndexed[attributes[attribute_with_most_gain]['index']] == value:
                    currentDict[value].append(instance)
        

        # print('attributes after the deletion', attributes)
        # print('')
        print('attribute with most gain', attribute_with_most_gain)

        attributes["num"] = attributes['num'] - 1


        for key, value in currentDict.items():
            print('here\'s what we\'re passing:')
            print('att w most gain', attribute_with_most_gain)
            print('key when we pass', key)
            print('s:', value)
            print('attributes:', attributes)
            print('classes:', classes)
            print('')

            Map<String, String[]> subMap = new HashMap<String, String[]>(map);
            attributes_left = attributes
            del attributes_left[attribute_with_most_gain]


            #mapPermute(subMap, currentPermutation + key + "=" + value + ", ");


            returnableDict.append(currentDict[key] = id3(value, attributes_left, classes))

        #finalDict = {attribute_with_most_gain: currentDict}
        
        return returnableDict

        


def get_most_gain(s, attributes, classes):
    
    keys = attributes.keys()
    gains = {}
    for key in keys:
        if key != 'num':
            gains[key] = get_gain(attributes[key]['values'], s, attributes[key]['index'], classes)
    
    #return the maximum of the gains
    return max(gains, key=gains.get)



def get_gain(variables, instances, index, classes):
    temp = [ [ 0 for y in range( len(classes) ) ] for x in range( len(variables) ) ]
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



#read in contents
file = open("fishingData.txt", "r")
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


final = id3(fileDictionary["s"], fileDictionary["attributes"], fileDictionary["classes"], {})

print(final)







#https://github.com/tofti/python-id3-trees/blob/master/id3.py



'''
def id3(S{}, ):
    if all examples in S are of the same class
        return a leaf with that class label
    else if there are no more attributes to test
        return a leaf with the most common class label
    else
        choose the attribute a that maximizes the Information Gain of S
        let attribute a be the decision for the current node
        add a branch from the current node for each possible value v of attribute a
        for each branch
            “sort” examples down the branches based on their value v of attribute a
            recursively call ID3(Sv) on the set of examples in each branch
'''

'''
totalNumData = int(data[7])

entropy = -(yesCount/totalNumData*math.log2(yesCount/totalNumData)+noCount/totalNumData*math.log2(noCount/totalNumData))
'''


