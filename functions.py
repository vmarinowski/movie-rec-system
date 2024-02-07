import pandas as pd
import yake
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from data import embeddings, movies, keywords
import transformers

sen_transformer = SentenceTransformer('all-mpnet-base-v2')

def get_len(row: str, plots: pd.DataFrame):
    length_all = 0 
    number_of_samples = len(plots)
    for delta in plots[row]:
        length_all += len(delta) #We add length of the each plot to 'length_all'.
    
    return round((length_all / number_of_samples))

#This function above will help us to demonstrate average length of the plots

def get_keywords(movies : pd.DataFrame):
    
    kw_extraction = yake.KeywordExtractor(lan = 'en', n = 3, top = 25)
    
    """
    lan - represents the language that you want to extract keyword.
    
    n - represents maximum n-gram length.
    
    top - number of keywords that will be extracted.
    
    """
    from tqdm import tqdm
    sentences = []
    total_iteration = len(movies.sample(25000))
    with tqdm(total= total_iteration) as pbar:
        for delta in movies['Plot']:
            texts = []
            keywords = kw_extraction.extract_keywords(delta)
            for keyword, _ in keywords:
                texts.append(keyword.lower())

            joined_text = ','.join(texts)
            sentences.append(joined_text)

            pbar.update(1)
    
    return sentences

#This function takes plot_index, embeddings, number of similar plots that you want to see and returns the index of similar movies in the dataset.

def get_similarity_score(plot_index, embeddings, num_of_similars):
    
    similarity = faiss.IndexFlatL2(768) #It uses L2 norm for calculating similarity 
    similarity.add(embeddings) 
    query_vector = embeddings[plot_index]
    
    #We will calculate similarity scores from all plots against query_vector.
    
    query_vector = query_vector.reshape(1, 768) #Adding 1 extra dimension to query vector.
    
    D, I = similarity.search(query_vector, num_of_similars) #This gives as most similar embeddings indexes with our query vector.
    
    return I

"""
This function is similar the function above, but this time it takes a plot our a message from user and print movies that similar the movie that
the user describe

"""

def get_similarity_from_plot(user_input, embeddings, num_of_similars, print_plot):
    
    kw_extractor = yake.KeywordExtractor(lan = 'en', n = 3, top = 25)
    similarity = faiss.IndexFlatL2(768)
    sen_transformer = SentenceTransformer('all-mpnet-base-v2')
    similarity.add(embeddings)
    text = []
    movies_reset_index = movies.reset_index(drop = True)
    keywords = kw_extractor.extract_keywords(user_input)
    
    
    
    for kw, _ in keywords:
        text.append(kw)
    
    joined_keywords = ','.join(text)
    input_embedding = sen_transformer.encode(joined_keywords)
    query_vector = input_embedding.reshape(-1, 768)
    
    #We are extracting keywords from user input and then convert them into embeddings.
    
    D, I = similarity.search(query_vector, num_of_similars)
    movie_titles = movies_reset_index.loc[I.reshape(-1), 'Title'].tolist()
    movie_plots = movies_reset_index.loc[I.reshape(-1), 'Plot'].tolist()
    rand_index = np.random.randint(num_of_similars, size = 1)
    
    #print(f'I recommend, {movie_titles}')
    
    if print_plot == True:
        print(f'Here is the plot:\n{movie_plots[rand_index[0]]}')

    movie_names = ''

    for x in movie_titles:
        movie_names = movie_names + x + ', '
    
    return movie_names[:-2]

#print(f'This is shape : {embeddings.shape}')