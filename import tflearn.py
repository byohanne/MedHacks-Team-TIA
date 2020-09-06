# Code Retrieved and adapted from https://youtu.be/u9FPqkuoEJ8

import tflearn 
import slurred_speech_data

learning_rate = 0.0001
training_iters = 300000

slurred_speech_batch = speech_data.mfcc_batch_generator(64)
X, Y = next(slurred_speech_batch)
trainX, trainY = X, Y
testX, testY, X, Y

#Using recurrent neural network capable of processing sequence of sound waves
# Makes use of Adam optimizer
net = tflearn.input_data([None, 20, 80])
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 10, activation="softmax")
net = tflearn.regression(net, optimizer="adam", learning_rate=learning_rate, loss="categorical_crossentropy")

model = tflearn.DNN(net, tensorboard_verbose=0)
while True:
    model.fit(trainX, trainY, n_epoch = 10, validation_set = (testX, testY), show_metric = True,
        slurred_speech_batch_size = 64)
    _y = model.predict(X)
model.save("tflearn.lstm.model")
print(_y)
print(y)


