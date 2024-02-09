# Movie Recommendation System
A movie recommendation web-app

You can test this app on -> [Streamlit](https://movie-rec-system-marinowski.streamlit.app)

# How this is work?
![algo](https://github.com/vmarinowski/movie-rec-system/assets/112823694/34037015-7450-44f6-9de6-5d7fe93b301c)

We take text samples as an input(in this case, it's the plots of the movies) and extract N number of keywords from each of them.(I used 25 in this notebook, because of the kaggle's limitations. I observed, the more is the better). This N keywords represents our whole plot. Then we convert these keywords into sentence embeddings using *Sentence Tokenizer*(There are bunch of methods which can turn words into embeddings, but I observed Sentence Tokenizer worked great). 

# Dataset
For this demo I was need a dataset, which contain both movie plots and movie titles. I found [this](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots) dataset, which contains decent amount of unique movies(and also their plot).

# Links
### [LinkedIn](https://www.linkedin.com/in/nureddin-aliyev-9b66b1274/)
### [Kaggle](https://www.kaggle.com/viceriomarinowski)
