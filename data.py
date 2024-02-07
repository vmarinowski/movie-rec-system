import pandas as pd
import numpy as np

df = pd.read_csv("data/embeddings.csv")
embeddings = df.values

movies = pd.read_csv("data/movies.csv")

with open("data/keywords.txt", 'r', encoding = 'utf-8') as text:
    keywords = [x.strip() for x in text.readlines()]
