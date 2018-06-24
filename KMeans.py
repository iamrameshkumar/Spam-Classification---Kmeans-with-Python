#import dependencies
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import gensim
import warnings
#hide runtime warnings
warnings.filterwarnings("ignore")
#load gensim model
fname = "text-8_gensim"
model = gensim.models.Word2Vec.load(fname)
print("Gensim model load complete...")
#read the csv file and drop unnecessary columns
df = pd.read_csv('./Youtube04-Eminem.csv', encoding="latin-1")
df = df.drop(['COMMENT_ID', 'AUTHOR', 'DATE'], axis=1)
original_df = pd.DataFrame(df)
df = df.drop(['CLASS'], axis=1)
#prepare the data in correct format for clustering
final_data = []
for i, row in df.iterrows():
    comment_vectorized = []
    comment = row['CONTENT']
    comment_all_words = comment.split(sep=" ")
    for comment_w in comment_all_words:
        try:
            comment_vectorized.append(list(model[comment_w]))
        except Exception as e:
            pass
    try:
        comment_vectorized = np.asarray(comment_vectorized)
        comment_vectorized_mean = list(np.mean(comment_vectorized, axis=0))
    except Exception as e:
        comment_vectorized_mean = list(np.zeros(100))
        pass
    try:
        len(comment_vectorized_mean)
    except:
        comment_vectorized_mean = list(np.zeros(100))
    temp_row = np.asarray(comment_vectorized_mean)
    final_data.append(temp_row)
X = np.asarray(final_data)
print('Conversion to array complete') 
print('Clustering Comments')
#perform clustering
clf = KMeans(n_clusters=2, n_jobs=-1, max_iter=50000, random_state=1)
clf.fit(X)
print('Clustering complete')
#If you want to save the pickle file for later use, uncomment the lines below
#joblib.dump(clf_news, './cluster_news.pkl')
#print('Pickle file saved')
#Put the cluster label in original dataframe beside CLASS label for comparison and save the csv file
comment_label = clf.labels_
comment_cluster_df = pd.DataFrame(original_df)
comment_cluster_df['comment_label'] = np.nan
comment_cluster_df['comment_label'] = comment_label
print('Saving to csv')
comment_cluster_df.to_csv('./comment_output.csv', index=False)
