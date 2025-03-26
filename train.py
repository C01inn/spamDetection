import csv, re, numpy as np, pickle
from tensorflow import keras
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def tokenize(emailData):
    tokenizer = Tokenizer(num_words=250, oov_token="<unk>")
    tokenizer.fit_on_texts(emailData)
    sequences = tokenizer.texts_to_sequences(emailData)
    paddedSequences = pad_sequences(sequences)
    return (paddedSequences, tokenizer)

# cleans text up
def cleanEmail(text) -> str:
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def createModel():
    model = keras.Sequential([
        Embedding(input_dim=250, output_dim=16),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')
    
    input2 = np.array(tokenizationLabels)

    model.fit(emails, input2, epochs=100)
    return model

def evaluateModel(model):
    loss, acc = model.evaluate(emails, np.array(tokenizationLabels))
    print(loss, acc)


emails = []
tokenizationLabels = []

# load training data
with open("training/trainingData.csv", 'r') as f:
    csvF = csv.DictReader(f)

    for line in csvF:
        # get email content and label
        tokenizationLabels.append(int(line["label_num"]))

        # preprocess email content here
        emails.append(cleanEmail(line["text"]))


emails, tokenizerr = tokenize(emails)
model = createModel()
evaluateModel(model)


# save tokenizer and model
model.save('modelData.keras')
with open('tokenizerData.keras', 'wb') as t:
    pickle.dump(tokenizerr, t, protocol=pickle.HIGHEST_PROTOCOL)