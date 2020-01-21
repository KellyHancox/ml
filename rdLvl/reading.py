import matplotlib.pyplot as plt
import numpy as np

papers = ["./3lp.txt", "./getty.txt", "./NYTimes.txt", "./mobyDick.txt", "./study.txt"]
paperTitles = ["3 pigs", "Gettysburg", "NYTimes", "Moby Dick", "Psych Study"]
papersRdLvl = []
vowels = ["a", "e", "i", "o", "u", "y"]
consonants = ["b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "q", "r", "s", "t", "v", "w", "x", "z"]
whitespace = ["\n", " "]
sentenceEnds = [".", "!", "?"]
punctuation = [",", ":", ";", "\""]
syllArray = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
syllCounts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
syllFileArray = []

numSentences = 0
numWords = 0
numSyllables = 0
prevChar = " "
syllInWord = 0
i = 0

for paper in papers:
    with open(paper) as f:
        while True:
            currChar = f.read(1)
            if not currChar:
                break
            
            if(currChar.isupper()):
                currChar = currChar.lower()

        #print(currChar)
            if(currChar in vowels):
                if ((prevChar in consonants) or (prevChar in whitespace)):
                    numSyllables += 1
                    syllInWord += 1
                    #print("syll")
            
            elif ((currChar in whitespace or currChar in punctuation) and (prevChar in vowels or prevChar in consonants)):
                if(prevChar is "e"):
                    numSyllables -= 1
                    syllInWord -= 1
                    #print("minusedSyll")

                numWords += 1
                #print("word")
                #print(syllInWord + 1)
                syllCounts[syllInWord - 1] += 1
                syllInWord = 0
            
            elif (currChar in sentenceEnds and ((prevChar in vowels) or (prevChar in consonants))):
                if(prevChar is "e"):
                    numSyllables -= 1
                    syllInWord -= 1
                    #print("minusedSyll")
            
                #in case of elipses or !!!
                if(prevChar not in vowels and prevChar not in consonants):
                    numSentences = numSentences - 1
                    #print("sentences minused")

                numSentences += 1
                numWords += 1

                syllCounts[syllInWord - 1] += 1
                syllInWord = 0
                #print("sentence")
                #print("word")
            

            prevChar = currChar

    '''
    print("num words: " + str(numWords))
    print("num sentences: " + str(numSentences))
    print("num syllables: " + str(numSyllables))
    '''

    #print("division: {}".format(numSyllables/numWords))

    fleschIndex = 206.835 - 84.6*(numSyllables/numWords) - 1.015*(numWords/numSentences)
    
    
    papersRdLvl.append(fleschIndex)
    #print(syllCounts)
    syllFileArray.append(syllCounts)
    syllCounts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


#print(fleschIndex)
print(syllFileArray)

fig, axs = plt.subplots(1, 5)

axs[0].bar(syllArray, syllFileArray[0])
axs[0].set_title(paperTitles[0])
axs[1].bar(syllArray, syllFileArray[1])
axs[1].set_title(paperTitles[1])
axs[2].bar(syllArray, syllFileArray[2])
axs[2].set_title(paperTitles[2])
axs[3].bar(syllArray, syllFileArray[3])
axs[3].set_title(paperTitles[3])
axs[4].bar(syllArray, syllFileArray[4])
axs[4].set_title(paperTitles[4])

fig.suptitle('Syllable Counts in Various Papers')
plt.tight_layout()
plt.show()


'''
print(papers)
print(papersRdLvl)

fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
axs[0].bar(paperTitles, papersRdLvl)
axs[1].scatter(paperTitles, papersRdLvl)
axs[2].plot(paperTitles, papersRdLvl)
fig.suptitle('Flesch Indexes by Papers')
plt.show()
'''
