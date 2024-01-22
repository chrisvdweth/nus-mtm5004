import os, re
import numpy as np
import random
import requests
import zipfile
import tarfile
import bz2
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from wordcloud import WordCloud
from scipy.cluster.hierarchy import dendrogram
from zipfile import ZipFile
from PIL import Image



# Turns a dictionary into a class
class Dict2Class():
    
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])
            

@staticmethod
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_line_count(file_name):
    try:
        with open(file_name, "rbU") as f:
            return sum(1 for _ in f)
    except:
        return None


def download_file(url, target_path='.', overwrite=False):
    # Preserve file name for download target
    file_name = url.split('/')[-1]
    if target_path.endswith('/'):
        path = target_path + file_name
    else:
        path = target_path + '/' + file_name
    # Check if file exists; only overwrite if specified
    if os.path.isfile(path) == True and overwrite is not True:
        print('File "{}" already exists.'.format(path))
        return path
    # Streaming, so we can iterate over the response
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    # Initialize progress bar
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    # Fetch file and write to path
    with open(path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    # Check for errors
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
        return None
    return path

        

def decompress_file(file_name, target_path='.', overwrite=False):
    if file_name.lower().endswith("zip"):
        zip_file = zipfile.ZipFile(file_name, 'r')
        zip_file.extractall(target_path)
        zip_file.close()
        return None
    elif file_name.lower().endswith("tar.gz"):
        tar = tarfile.open(file_name, "r:gz")
        tar.extractall(path=target_path)
        tar.close()
        return None
    elif file_name.lower().endswith("tar"):
        tar = tarfile.open(file_name, "r:")
        tar.extractall(path=target_path)
        tar.close()
        return None
    elif file_name.lower().endswith("bz2"):
        output_file_name = target_path + file_name.split('/')[-1]
        output_file_name = re.sub('.bz2', '', output_file_name, flags=re.I)
        # Check if file exists; only overwrite if specified
        if os.path.isfile(output_file_name) == True and overwrite is not True:
            print('File "{}" already exists.'.format(output_file_name))
            return output_file_name
        with open(output_file_name, 'wb') as output_file, bz2.BZ2File(file_name, 'rb') as file:
            for data in iter(lambda : file.read(100 * 1024), b''):
                output_file.write(data)
        return output_file_name
        
        
        
def plot_training_results(results):
    
    x = list(range(1, len(results)+1))
    
    losses = [ tup[0] for tup in results ]
    acc_train = [ tup[1] for tup in results ]
    acc_test = [ tup[2] for tup in results ]

    # Convert losses to numpy array
    losses = np.asarray(losses)
    # Normalize losses so they match the scale in the plot (we are only interested in the trend of the losses!)
    losses = losses/np.max(losses)

    plt.figure()

    plt.plot(x, losses, lw=3)
    plt.plot(x, acc_train, lw=3)
    plt.plot(x, acc_test, lw=3)

    font_axes = {'family':'serif','color':'black','size':16}

    plt.gca().set_xticks(x)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.xlabel("Epoch", fontdict=font_axes)
    plt.ylabel("F1 Score", fontdict=font_axes)
    plt.legend(['Loss', 'F1 (train)', 'F1 (test)'], loc='lower left', fontsize=16)
    plt.tight_layout()
    plt.show()

    
    
def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=[color]*len(x), alpha=a, label=label, s=100)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.8, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=16)
    plt.legend(loc=4, fontsize=12)
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.show()    

    
def plot_attention_weights(weights, src_labels, tgt_labels):
    plt.figure(figsize=(10,10))
    hm = sns.heatmap(weights,
                     cbar=True,
                     annot=True,
                     square=True,
                     fmt='.2f',
                     annot_kws={'size': 10},
                     yticklabels=tgt_labels,
                     xticklabels=src_labels)
    plt.title('Attention Weights')
    plt.tight_layout()
    plt.show()    
    
    
    
def get_articles(dataset_file, search_term=None):    
    articles = []
    
    input_zip = ZipFile(dataset_file)
    for file_name in input_zip.namelist():
        with input_zip.open(file_name) as file:
            for line in file:
                line = line.strip()
                article = "{}".format(line.decode("utf-8"))
                
                if search_term is not None and re.search(r"{}".format(search_term), article, flags=re.IGNORECASE) is None:
                    continue
                
                articles.append(article)

    return articles



def get_random_article(dataset_file):
    articles = []
    
    input_zip = ZipFile(dataset_file)
    for file_name in input_zip.namelist():
        lines = input_zip.open(file_name).read().splitlines()
        return random.choice(lines).decode("utf-8")

    
    
def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(210, 100%, 24%)"


def get_mask():
    return np.array(Image.open("data/images/oval-mask.png"))


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
    

def compute_sparsity(M):
    return M[M==0].size / M.size
