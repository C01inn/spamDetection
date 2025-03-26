import pickle, json
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences


global tokenizer
global model
global samples


with open('tokenizerData.keras', 'rb') as t:
    tokenizer = pickle.load(t)    

model = keras.models.load_model('modelData.keras')


def predict(texts):

    sequences = tokenizer.texts_to_sequences(texts)
    texts = pad_sequences(sequences)

    preditions = model.predict(texts)

    # round to 0 or 1
    c = []
    for x in list(preditions):   
        floatNum = list(x)[0].item()
        c.append(round(floatNum)) 
        
    return c



def determineAccuracy(a, b):
    if len(a) != len(b):
        print("Cannot calculate accuracy")
        return

    count = 0
    for idx, i in enumerate(a):
        if i == b[idx]:
            count+=1
    print("\nAI Model Accuracy: " + str((count / len(a)) * 100))


with open('samples/samples.json', 'r') as f:
    samples = json.load(f)

textSamples = []
correctResponses = []

for x in samples:
    textSamples.append(x[0])
    correctResponses.append(x[1])

predictions = predict(textSamples)
determineAccuracy(predictions, correctResponses)