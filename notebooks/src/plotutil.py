import numpy as np
import matplotlib.pyplot as plt

from itertools import cycle
from wordcloud import WordCloud
from scipy.cluster.hierarchy import dendrogram
from PIL import Image

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


colors = 'bgrcmykbgrcmykbgrcmykbgrcmyk'


def show_wordcloud(source, max_words=50):
    try:
        wordcloud = WordCloud(max_words=max_words)
        if isinstance(source, str):
            wordcloud.generate_from_text(source)
        else:
            wordcloud.generate_from_frequencies(source)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
    except:
        raise ValueError("Invalid data type for source parameter: str or [(str,float)]")
           
            
            
def show_clusters_high_dim(model, X, method='pca', title=''):
    method = method.lower().strip()
    if method == 'pca':
        reduced_data = PCA(n_components=2).fit_transform(X)
    elif method == 'tsne':
        reduced_data = TSNE(n_components=2).fit_transform(X)
    else:
        raise ValueError("Invalid data type for method parameter: 'pca' or 'tsne'")
    # Get the labels from model
    labels = model.labels_
    # set up plot
    fig, ax = plt.subplots(figsize=(17, 9)) # set size
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

    for idx, instance in enumerate(reduced_data):
        label = labels[idx]
        color = colors[label]
        ax.plot(instance[0], instance[1], marker='o', color=color, linestyle='', ms=15, mec='none')
        ax.set_aspect('auto')
    
    plt.show() #show the plot


    
def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(210, 100%, 24%)"


def get_mask():
    return np.array(Image.open("data/images/div/oval-mask.png"))

    
def plot_sse_values(sse):
    plt.figure()
    plt.xlabel('K', fontsize=16)
    plt.ylabel('SSE', fontsize=18)
    plt.tick_params(axis="x", labelsize=12)
    plt.tick_params(axis="y", labelsize=12)
    plt.plot([s[0] for s in sse], [s[1] for s in sse], marker='o', lw=2)
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.tight_layout()
    plt.show()
    
    
def plot_silhouette_scores(silhouette_scores):
    plt.figure()
    plt.xlabel('K', fontsize=16)
    plt.ylabel('SC', fontsize=18)
    plt.tick_params(axis="x", labelsize=12)
    plt.tick_params(axis="y", labelsize=12)
    plt.plot([s[0] for s in silhouette_scores], [s[1] for s in silhouette_scores], marker='o', lw=2)
    plt.tight_layout()
    plt.show()    
    
    
    
def plot_cluster_wordcloud(vectorizer, kmeans, label):
    # Get indices of all articles in the same cluster
    #article_indices = np.argwhere(clustering==label)
    # Extract the respective articles    
    #cluster_articles = [ article for idx, article in enumerate(articles) if idx in article_indices ]
    
    centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    
    term_frequencies = kmeans.cluster_centers_[label]
    sorted_terms = centroids[label]
    frequencies = {terms[i]: term_frequencies[i] for i in sorted_terms}
    
    wc = WordCloud(color_func=color_func, background_color="white", max_words=500, mask=get_mask(), contour_width=0)
    #wc = WordCloud(background_color="white", max_words=50)
    # generate word cloud
    wc.generate_from_frequencies(frequencies)

    # show
    plt.figure()
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.show()    
    
    
def plot_dendrogram(model, labels, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    plt.figure()
    dendrogram(linkage_matrix, labels=labels, **kwargs)
    plt.xticks(rotation=65, ha='right')
    plt.tight_layout()
    plt.show()
