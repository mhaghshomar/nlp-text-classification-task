from sklearn import preprocessing, metrics
from sklearn.neural_network import MLPClassifier
from nltk.tokenize import word_tokenize
import csv
import random
with open('StrictOMD.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

words_list = []
inputs = []
labels = []


for i in data:
    tokenized_tweets = word_tokenize(i[1])
    words_list = words_list + tokenized_tweets

frequency_dictionary = {}
unique_word_count = 0
unique_word_list = []

for word in words_list:
    if word in frequency_dictionary.keys():
        frequency_dictionary[word] += 1
    else:
        frequency_dictionary[word] = 1
        unique_word_count += 1
        unique_word_list.append(word)

for i in data:
    tokenized_tweets = word_tokenize(i[1])
    input = []
    for j in unique_word_list:
        count = 0
        for k in tokenized_tweets:
            if j == k:
                count += 1
        input.append(count)
    inputs.append(input)
    labels.append(i[0])

training_inputs = []
training_labels = []
test_inputs = inputs
test_labels = labels

for i in range(len(inputs)//4):
    j = random.randint(0, len(inputs) - 1)
    training_inputs.append(inputs[j])
    training_labels.append(labels[j])
    del test_inputs[j]
    del test_labels[j]

training_inputs = preprocessing.StandardScaler().fit_transform(training_inputs)
classifier = MLPClassifier(max_iter=1500, hidden_layer_sizes=(60, 15), random_state=5)

classifier.fit(training_inputs, training_labels)
y_prediction = classifier.predict(test_inputs)


print(metrics.accuracy_score(test_labels, y_prediction))




