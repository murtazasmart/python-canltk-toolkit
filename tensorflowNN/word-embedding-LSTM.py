
import fasttext

import tensorflow as tf

import tensorflow_datasets as tfds

# Imports the chat and trains it

INPUT_FILE_NAME = 'D:\MSc\Chat Parser Script\chat-data\processed-chat\chat3-AbbasJafferjee-HamzaNajmudeen.txt'
# model = fasttext.train_unsupervised(INPUT_FILE_NAME, model='skipgram',dim=500)
# print(model.words)
# vocab_size = len(model.words)
# print(vocab_size)

# print(len(model.get_sentence_vector("Try and check out out..  There are like more than 150. Ppl")))
# print(len(model.get_sentence_vector(":p")))

# print((model.get_sentence_vector("Try and check out out..  There are like more than 150. Ppl")))


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')

# STEP 2 - This sets up the test and train data
data = []

def labeler(example, index):
    return example, tf.cast(index, tf.int64)  

INPUT_FILE_NAME1 = "D:\MSc\Chat Parser Script\chat-data\processed-chat\chat3-AbbasJafferjee-HamzaNajmudeen.txt"
lines_dataset1 = tf.data.TextLineDataset(INPUT_FILE_NAME1)
labeled_dataset1 = lines_dataset1.map(lambda ex: labeler(ex, 1))

INPUT_FILE_NAME2 = "D:\MSc\Chat Parser Script\chat-data\extracted-features\chat3-inverse-AbbasJafferjee-HamzaNajmudeen.txt"
lines_dataset2 = tf.data.TextLineDataset(INPUT_FILE_NAME2)
labeled_dataset2 = lines_dataset2.map(lambda ex: labeler(ex, 0))

for ex in labeled_dataset2.take(5):
  print(ex)

dataset = labeled_dataset1.concatenate(labeled_dataset2)

# STEP 3 - Carries out the encoding using trained model
test_encoding = model.encode("Crazy")
print("LLO")
print(len(test_encoding))
test_encoding = model.encode("Tis women is hectic men.")
print("LLO")
print(len(test_encoding))
count = 0
enc_list = []
enc_dict = {}
for ex in labeled_dataset1:
    # vec = model.get_sentence_vector(str(ex[0].numpy()))
    vec = model.encode(str(ex[0].numpy()))
    enc_list.append(vec)
    enc_dict[ex[0].numpy()] = vec
for ex in labeled_dataset2:
    # vec = model.get_sentence_vector(str(ex[0].numpy()))
    vec = model.encode(str(ex[0].numpy()))
    enc_list.append(vec)
    enc_dict[ex[0].numpy()] = vec

print(len(enc_list))

def encode(text_tensor, label):
    encoded_text = enc_dict[text_tensor.numpy()]
    return encoded_text, label

def encode_map_fn(text, label):
    # py_func doesn't set the shape of the returned tensors.
    encoded_text, label = tf.py_function(encode, 
                                    inp=[text, label], 
                                    Tout=(tf.float64, tf.int64))

    # `tf.data.Datasets` work best if all components have a shape set
    #  so set the shapes manually: 
    encoded_text.set_shape([None])
    label.set_shape([])

    return encoded_text, label

# example_text = next(iter(dataset))[0].numpy()
# print(example_text)

# encoded_example = model.get_word_vector(example_text)
# print(encoded_example)

encoded_dataset = dataset.map(encode_map_fn)

# sample_text, sample_labels = next(iter(encoded_dataset))

# print(sample_text[0], sample_labels)

# STEP 4 - Setup and Train the LSTM
BUFFER_SIZE = 100
BATCH_SIZE = 64
TAKE_SIZE = 50
# dataset = tf.data.Dataset.from_tensor_slices(data)
encoded_dataset = encoded_dataset.shuffle(buffer_size=BUFFER_SIZE)

train_data = encoded_dataset.skip(TAKE_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE)

test_data = encoded_dataset.take(TAKE_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE)

# # for ex in train_data.take(5):
# #     print(ex)
# sample_text, sample_labels = next(iter(test_data))

# print(sample_text[0], sample_labels[0])

look_back = 3
num_features = 

nb_samples = X_train.shape[0] - look_back

x_train_reshaped = np.zeros((nb_samples, look_back, num_features))
y_train_reshaped = np.zeros((nb_samples))

for i in range(nb_samples):
    y_position = i + look_back
    x_train_reshaped[i] = X_train[i:y_position]
    y_train_reshaped[i] = y_train[y_position]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units, activation='relu'))
# model.add(tf.keras.layers.Embedding(len(test_encoding), 64))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
# model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
# One or more dense layers.
# Edit the list in the `for` line to experiment with layer sizes.
for units in [64, 64]:
  model.add(tf.keras.layers.Dense(units, activation='relu'))

# Output layer. The first argument is the number of labels.
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn')])

model.fit(train_data, epochs=100, validation_data=test_data)

loss, accuracy, tp, fp, tn, fn,   = model.evaluate(test_data)
print('Test Loss: {}'.format(loss))
print('Test Accuracy: {}'.format(accuracy))
print('Test TP: {}'.format(tp))
print('Test FP: {}'.format(fp))
print('Test TN: {}'.format(tn))
print('Test FN: {}'.format(fn))

# print('\nEval loss: {:.3f}, Eval accuracy: {:.3f}'.format(eval_loss, eval_acc))
