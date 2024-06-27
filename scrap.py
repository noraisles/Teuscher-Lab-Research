import re
#example file: Sentiment analysis notes - condensed (1).txt
def calculate(cleaned_text_three, afinn):
    global word
    words = re.findall(r'\b\w+\b',text.lower())
    sentiment_score = sum(afinn.get(word, 0) for word in words)
    return sentiment_score
def file():
    global filename
    with open(filename, 'r') as file:
        text = file.read()
        return text

while format != "t" and format != "f":
    format = input("Would you like to write the text or enter a file? (t/f) ")
    if format == "t":
        text = input("Please enter the phrase you would like to test: ")
    if format == "f":
        filename = input("What is the name of your file (include .txt)? ")
        with open(filename, 'r') as file:
            text = file.read()

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

