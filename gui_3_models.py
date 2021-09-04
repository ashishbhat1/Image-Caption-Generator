from tkinter import *
from tkinter import ttk

from PIL import Image, ImageTk
from tkinter import filedialog

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import InceptionV3, Xception
from tensorflow.keras.applications.inception_v3 import preprocess_input
# from keras.layers.merge import add
from tensorflow.keras.layers import GlobalAveragePooling1D, LayerNormalization, Bidirectional, LSTM, Dense, RepeatVector, Embedding, Dropout, Activation, Input, BatchNormalization, GlobalAveragePooling2D, Flatten, Layer, concatenate, Lambda, Add, dot, MaxPooling1D, Conv1D, Multiply, GRU
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam, SGD, Adadelta
import tensorflow as tf
import numpy as np
import pandas as pd
import re
import pickle

max_len = 25
embdim = 300
nheads = 10
dff = 512
rate = 0.2


class MultiHeadAttention(Layer):
    def __init__(self, embdim, nheads):
        super(MultiHeadAttention, self).__init__()
        self.embdim = embdim
        self.nheads = nheads
        self.wq = Dense(embdim)
        self.wk = Dense(embdim)
        self.wv = Dense(embdim)
        self.out_dense = Dense(embdim)
        self.depth = embdim//nheads

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.nheads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_attention(self, q, k, v):
        # k=(tf.math.square(k))
        # q=(tf.math.square(q))
        qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_qk = qk/tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_qk, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights

    def call(self, inputs):
        q = self.wq(inputs)
        v = self.wv(inputs)
        k = self.wk(inputs)
        batch_size = tf.shape(inputs)[0]
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, weights = self.scaled_dot_attention(q, k, v)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.embdim))
        output = self.out_dense(concat_attention)
        return output


class PositionalEncoding(Layer):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

    def call(self, inputs):
        position = tf.shape(inputs)[1]

        position_dims = tf.range(position)[:, tf.newaxis]
        embed_dims = tf.range(self.d_model)[tf.newaxis, :]
        angle_rates = 1 / tf.pow(
            10000.0, tf.cast(
                (2 * (embed_dims // 2)) / self.d_model, tf.float32))
        angle_rads = tf.cast(position_dims, tf.float32) * angle_rates

        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)


class Transformer(Layer):
    def __init__(self, embdim, nheads, dff, rate, **kwargs):
        super(Transformer, self).__init__()
        self.ffn1 = Dense(dff, activation='relu')
        self.ffn3 = Dense(dff/2, activation='relu')
        self.ffn4 = Dense(dff/4, activation='relu')
        self.ffn2 = Dense(embdim)
        self.mha = MultiHeadAttention(embdim, nheads)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)
        self.dropout4 = Dropout(rate)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'embdim': embdim,
            'nheads': nheads,
            'dff': dff,
            'rate': rate,
        })
        return config

    def call(self, inputs):
        att_out = self.mha(inputs)
        att_out = self.dropout1(att_out)
        out1 = self.layernorm1(inputs+att_out)
        ff = self.ffn1(out1)
        ff = self.ffn2(ff)
        ff = self.dropout2(ff)
        return self.layernorm2(out1+ff)


class attention(Layer):
    def __init__(self, dim):
        super(attention, self).__init__()
        self.W = Dense(dim)
        self.W2 = Dense(dim)
        self.W3 = Dense(1)

    def call(self, features, prev):
        prev = tf.expand_dims(prev, 1)
        prev = self.W2(prev)
        c = self.W(features)
        c = tf.math.tanh(c+prev)
        score = self.W3(c)
        # print(c.shape)
        aw = tf.nn.softmax(score, axis=1)
        q = aw*features
        q = tf.reduce_sum(q, axis=1)
        return q


# model1 = tf.keras.models.load_model(
#     './image_cap_model_flickr40k.h5', custom_objects={'Transformer': Transformer})

model2 = tf.keras.models.load_model(
    './image_cap_model_transformer_flickr30k.h5', custom_objects={'Transformer': Transformer})

model3 = tf.keras.models.load_model('image_cap_model_lstm_flickr30k.h5')

with open('./tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
index_word = dict([(index, word)
                   for word, index in tokenizer.word_index.items()])


def greedySearch(img_path):
    in_text = 'startseq'
    # img_path = '../input/flickr-image-dataset/flickr30k_images/flickr30k_images/' + \
    #     str(img_path)
    im = load_img(img_path, target_size=(224, 224, 3))
    im = img_to_array(im)
    photo = preprocess_input(im)
    photo = photo.reshape((1, 224, 224, 3))
    for i in range(max_len):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_len)
        yhat = (model2.predict([photo, sequence], verbose=0) +
                model3.predict([photo, sequence], verbose=0))/2
        yhat = np.argmax(yhat)
        word = index_word[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


root = Tk()
root.title("Title")
root.geometry('1450x800')


def upload_image(event=None):
    destroy_frame()
    l2 = Label(bg_label, text='UPLOADED IMAGE:', bg='cyan',
               width=15, height=2, font=('courier', 18))
    l2.place(relx=0.1, rely=0.1)
    global uploaded_image
    global caption
    img_filename = filedialog.askopenfilename()
    caption = greedySearch(str(img_filename))
    img = Image.open(img_filename)
    img = img.resize((500, 300), Image.ANTIALIAS)

    print(caption)
    uploaded_image = ImageTk.PhotoImage(img)
    uploaded_label = Label(bg_label, image=uploaded_image)
    uploaded_label.place(relx=0.35, rely=0.2)

    b2 = Button(bg_label, text='GENERATE A CAPTION',
                width=50, height=3, command=pg2)
    b2.place(relx=0.4, rely=0.75)
    print('Selected: ', img_filename)


def pg2():
    destroy_frame()
    uploaded_label = Label(bg_label, image=uploaded_image)
    uploaded_label.place(relx=0.35, rely=0.15)

    l3 = Label(bg_label, text=caption,
               fg='black', font=('courier', 25))
    l3.place(relx=0.2, rely=0.75)
    button_quit = Button(root, text='Exit Program', command=root.quit)
    button_quit.place(relx=0.9)
    button_return = Button(root, text='Upload', command=upload_image)
    button_return.place(relx=0.1)


def resize_image(event):
    new_width = event.width
    new_height = event.height
    image = copy_of_image.resize((new_width, new_height))
    photo = ImageTk.PhotoImage(image)
    bg_label.config(image=photo)
    bg_label.image = photo


def destroy_frame(event=None):
    for widget in bg_label.winfo_children():
        print(widget)
        widget.destroy()


image = Image.open('projbg.png')
copy_of_image = image.copy()
photo = ImageTk.PhotoImage(image)
bg_label = ttk.Label(root, image=photo)
bg_label.bind('<Configure>', resize_image)
bg_label.pack(fill=BOTH, expand=YES)


b1 = Button(bg_label, text='UPLOAD IMAGE', background='gray',
            width=50, height=3, command=upload_image)
b1.place(x=500, y=600)

l1 = Label(bg_label, text='GENERATE A CAPTION FOR AN IMAGE',
           bg='gray', fg='cyan', font=('courier', 48))
l1.place(relwidth=1, rely=0.1)


root.mainloop()
