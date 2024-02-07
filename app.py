import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import transformers
import faiss #For calculating similarity between embeddings. 
from sentence_transformers import SentenceTransformer #For Sentence Embeddings.
from data import embeddings
from functions import get_similarity_from_plot

embeddings = embeddings[:, 1:]

st.image('./header.jpg')

st.header('Movie Recommendation System')



num_of_similars = st.slider(label = 'How many recommendation do you want?',
            min_value = 1,
            max_value = 5)

text = st.text_area(label = "Describe the movie that you'd like to watch.")

button = st.button(label = 'Ask')


if button == True:

    output = get_similarity_from_plot(user_input = text, embeddings = embeddings,
                            num_of_similars = num_of_similars, print_plot = False)
    st.write('You should watch: ', output)

with st.sidebar:
    st.header('What is this?')
    st.write("""Watching a movie is the most peaceful thing for me. I love movies, once upon a time I wanted to become a movie Director or Screenplay writer, but I couldn't (at least for now). Even I couldn't become a Director these flames in my heart haven't gone out. I wanted to somehow useful, what could I do? Then the idea came up to my mind. You know, when your friends talking about a movie and they ask any other movie recommendation which, similar to that movie. As an intellectual being you thinking about its plot and then compare its plot with similar movies which you watched. Eventually you come up with a recommendation. I want from Machine to make this type of recommendation to people instead of you.
             """)
    st.link_button('Source code', 'www.youtube.com')

