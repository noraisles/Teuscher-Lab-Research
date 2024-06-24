import re

def calculate(cleaned_text_three, afinn):
    global word
    words = re.findall(r'\b\w+\b',text.lower())
    sentiment_score = sum(afinn.get(word, 0) for word in words)
    return sentiment_score


text = input("Please enter the phrase you would like to test: ")
cleaned_text_one = text.replace("n't", " not")
cleaned_text_two = re.sub(r'[^\w\s]', '', cleaned_text_one)
cleaned_text_three = cleaned_text_two.lower()
print(cleaned_text_three)
afinn = {}

with open('originalAFINN.txt') as file:
    for line in file:
        word, score = line.split('\t')
        afinn[word] = int(score)

score = calculate(cleaned_text_three, afinn)
print("Sentiment score: {}".format(score))
