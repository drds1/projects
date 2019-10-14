import pandas as pd

def load_sentiment_data(path = './input/sentiment labelled sentences',
                        file_dict={'yelp': 'yelp_labelled.txt',
                                   'amazon': 'amazon_cells_labelled.txt',
                                   'imdb': 'imdb_labelled.txt'}):
    '''
    load the sentiment data on comments from amazon
    :return:
    '''
    df_list = []
    for source, file in file_dict.items():
       df = pd.read_csv(path+'/'+file, names=['sentence', 'label'], sep='\t')
       # Add another column filled with the source name
       df['source'] = source
       df_list.append(df)

    df = pd.concat(df_list)
    print(df.head())
    return df
