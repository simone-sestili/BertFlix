import os
import glob
import json
import re
import time
import torch
import zipfile
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from datetime import datetime
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, CrossEncoder, util, InputExample, losses, evaluation

from utils.utils import load_pickle, dump_pickle
from utils.opt import fair_sequence_division_order



# ========== PRE-PROCESSING ==========

def pipeline(text: str, strings_to_exclude: list = []) -> str:
    """
    Clean given line from special characters
    """
    # remove lines with only dashes
    if re.match(r"^\s*-+\s*$", text):
        return ''
    # remove lines with pip install
    if re.match(r"pip install .+?", text):
        return ''
    # remove html tags
    text = re.sub(r"<.+?>", '', text)
    # remove urls
    text = re.sub(r"http\S+", '', text)
    # remove chinese characters
    text = re.sub(r"[\u4e00-\u9fff]+", '', text)
    
    text = text.lower().strip()
    
    # remove useless strings
    for excl in strings_to_exclude:
        if text.startswith(excl):
            return ''

    return text


def processing(text: str) -> str:
    # split text to reduce size
    # sub_paragraphs = text_length_reduction(text, target_words, '\n')
    return text.lower().strip('\n').strip()



# ========== LOAD FUNCTIONS ==========

def load_data(data_folder: str, filename: str, download_url: str = ''):
    """
    Load data in filename, if it needs to be downloaded it creates the folder, download the file and load/unzip the result.
    If the result is not a structured object for text (image scenario) return the path of the folder containing the images.
    """
    data_file_path = os.path.join(data_folder, filename)

    # creates data folder
    if not os.path.exists(data_folder):
        os.makedirs(data_folder, exist_ok=True)

    # download from web
    if download_url:
        # check if data is already present
        if not os.path.exists(data_file_path):
            util.http_get(download_url, data_file_path)
    
    # decide load method depending on extension
    ext = data_file_path.split('.')[-1]
    
    if ext == 'json':
        with open(data_file_path, encoding='utf-8') as f:
            json_data = json.load(f)
        print(f'Loaded {len(json_data)} items')
        return pd.DataFrame(json_data)
    
    elif ext == 'csv':
        df = pd.read_csv(data_file_path)
        print(f'Loaded {len(df)} items')
        return df
    
    elif ext == 'zip':
        unzip_data_path = '.'.join(data_file_path.split('.')[:-1])
        if not os.path.exists(unzip_data_path):
            # unzip data
            os.makedirs(unzip_data_path)
            with zipfile.ZipFile(data_file_path, 'r') as zf:
                for member in tqdm(zf.infolist(), desc='Extracting'):
                    zf.extract(member, unzip_data_path)
        print(f'Loaded {len(os.listdir(unzip_data_path))} items')
        return unzip_data_path
    else:
        print(f'File {filename} is not currently supported')
        return None
    
    
def load_model(model_folder: str, model_name: str):
    """
    If model is in local files load it from there, otherwise download from HuggingFace
    """
    if not os.path.exists(model_folder):
        os.makedirs(model_folder, exist_ok=True)

    if model_name.startswith('cross-encoder'):
        # cross-encoder
        if model_name not in os.listdir(model_folder):
            model = CrossEncoder(model_name)
            model.save(os.path.join(model_folder, model_name))
        else:
            model = CrossEncoder(os.path.join(model_folder, model_name))
        return model
    else:
        # bi-encoder
        if model_name not in os.listdir(model_folder):
            model = SentenceTransformer(model_name)
            model.save(os.path.join(model_folder, model_name))
        else:
            model = SentenceTransformer(os.path.join(model_folder, model_name))
        return model


def load_embeddings(data, model, embeddings_path: str, use_precomputed: bool = False, download_url: str = '', batch_size: int = 32, data_type: str = 'text'):
    """
    This function loads and returns the aligned couple corpus_names, corpus_embeddings.
    - If user decides to use the precomputed embeddings the functions tries to load them from local storage or 
      from web, if this process fails it goes to the manual creation.
    - If embeddings must be manually created the process depends on the data_type, which can be either text or image. If the data is text based
      it has to be passed as a list of strings, it the data is image based it has to be passed the path of folder containing all the images.
    """
    if use_precomputed:
        if os.path.exists(embeddings_path):
            # load local embeddings
            corpus_names, corpus_embeddings = load_pickle(embeddings_path)
            print('Items:', len(corpus_names))
            return corpus_names, corpus_embeddings
        elif download_url:
            # download from web and store into embeddings_path
            util.http_get(download_url, embeddings_path)
            try:
                corpus_names, corpus_embeddings = load_pickle(embeddings_path)
                print('Items:', len(corpus_names))
                return corpus_names, corpus_embeddings
            except:
                pass  # something went wrong during download, re-compute embeddings
    if data_type == 'text' and type(data) == list:  # assume data is an ordered list of strings
        corpus_names = data
        print('Paragraphs:', len(corpus_names))
        corpus_embeddings = model.encode(
            data,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=True
        )
    elif data_type == 'image' or type(data) == str:  # assume data is images path
        corpus_names = list(glob.glob(data + '/*'))
        print('Images:', len(corpus_names))
        corpus_embeddings = model.encode(
            [Image.open(filepath) for filepath in corpus_names],
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=True
        )
    dump_pickle((corpus_names, corpus_embeddings), embeddings_path)
    return corpus_names, corpus_embeddings



# ========== TASKS IMPLEMENTATION ==========

def image_clustering(corpus_names: list, corpus_embeddings: list, folder_file_path: str, threshold: float = 0.75, min_community_size: int = 10, clusters_to_show: int = 10, results_to_show: int = 3):
    """
    Fast method to create clusters of images with similar meaning.
    The threshold value controls the selectivity to differentiate different clusters.
    Shows in a jupyter notebook some images from some clusters.
    """
    from IPython.display import display
    from IPython.display import Image as IPImage
    
    clusters = util.community_detection(corpus_embeddings, threshold=threshold, min_community_size=min_community_size)
    for cluster in clusters[:clusters_to_show]:
        print("\n\nCluster size:", len(cluster))
        # output first 3 images
        for idx in cluster[:results_to_show]:
            display(IPImage(os.path.join(folder_file_path, corpus_names[idx]), width=200))
        time.sleep(0.1)
    
    return clusters


def image_classification(corpus_names: list, corpus_embeddings: list, folder_file_path: str, labels: list, model, results_to_show: int = 10):
    """
    Classified a corpus of images converted into word embeddings within a given set of labels.
    Shows a given number of results in a jupyter notebook.
    """
    from IPython.display import display
    from IPython.display import Image as IPImage
    
    # convert labels to word embeddings
    labels_embeddings = model.encode(labels, convert_to_tensor=True)
    
    # compute cross scores through cosine similarity
    cos_scores = util.cos_sim(corpus_embeddings, labels_embeddings)
    
    # extracts highest cosine similarity for each image
    pred_labels = torch.argmax(cos_scores, dim=1)
    
    # show results
    for idx in range(results_to_show):
        display(IPImage(os.path.join(folder_file_path, corpus_names[idx]), width=200))
        print(f'Predicted label: {labels[pred_labels[idx]]}\n\n')
        time.sleep(0.1)
    
    return [labels[pred] for pred in pred_labels]


def text2image(query: str, image_names: list, image_embeddings: list, folder_file_path: str, img_model, top_k: int = 5):
    """
    Given a textual query it return a set of images with the most similar meaning
    """
    from IPython.display import display
    from IPython.display import Image as IPImage
    
    # convert query to word embedding using CLIP
    query_embedding = img_model.encode([query], convert_to_tensor=True, show_progress_bar=False)
    
    # search most similar images
    results = util.semantic_search(query_embedding, image_embeddings, top_k=top_k)
    results = results[0]  # get result of the first query
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    
    # show results
    print('Query:')
    display(query)
    for res in results:
        print('Score', round(res['score'], 3))
        display(IPImage(os.path.join(folder_file_path, image_names[res['corpus_id']]), width=200))
        time.sleep(0.1)
    
    return results


def text_clustering(df: pd.DataFrame, corpus_embeddings: list, threshold: float = 0.75, min_community_size: int = 10, clusters_to_show: int = 10, results_to_show: int = 3, cols_to_show: list = ['text']):
    """
    Fast method to create clusters of texts with similar meaning.
    The threshold value controls the selectivity to differentiate different clusters.
    Shows in a jupyter notebook some texts from some clusters. 
    """
    clusters = util.community_detection(corpus_embeddings, threshold=threshold, min_community_size=min_community_size)
    for cluster in clusters[:clusters_to_show]:
        print("\n\nCluster size:", len(cluster))
        # output first 3 texts
        for idx in cluster[:results_to_show]:
            print('->', idx)
            for col in cols_to_show:
                print(df.iloc[idx][col])
            print('-'*50)
        time.sleep(0.1)
    return clusters


def text_search(query: str, df: pd.DataFrame, corpus_embeddings: list, model, top_k: int = 10, results_to_show: int = 5, cols_to_show: list = ['text']):
    """
    Given a text query performs a semantic search on a given set of text, already converted in word embeddings.
    """
    # convert query to a word embedding
    query_embedding = model.encode(query, convert_to_tensor=True)
    query_embedding = query_embedding.cpu()
    
    # compare to given embedding through cosine similarity
    results = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
    results = results[0]  # get results of the first query
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    
    # show results
    for res in results[:results_to_show]:
        print('Score', round(res['score'], 3))
        for col in cols_to_show:
            print(df.iloc[res['corpus_id']][col])
    
    return results


def text_rerank(query: str, df: pd.DataFrame, corpus_names: list, search_hits: list, model, results_to_show: int = 5, cols_to_show: list = ['text']):
    """
    Given a query and some results obtained with a bi-encoder semantic search improves the results
    by re-ranking them through a cross-encoder that directly compares the query with all the previous results.
    """
    # create list of couples to compare
    couples_to_compare = [[query, corpus_names[hit['corpus_id']]] for hit in search_hits]
    
    # compute scores through cross-encoder and update search_hits
    cross_scores = model.predict(couples_to_compare)
    
    results = search_hits
    for idx, score in enumerate(cross_scores):
        results[idx]['score'] = cross_scores[idx]
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    
    # show results
    for res in results[:results_to_show]:
        print('Score', round(res['score'], 3))
        for col in cols_to_show:
            print(df.iloc[res['corpus_id']][col])
    
    return results


def date_reranking(hits: list, df: pd.DataFrame, max_increment: float, slope: float, step: str = 'year') -> list:
    """
    Updates the scores in the list of the search hits by adding a value that it higher the closer each hits is to the current date.
    When the hit is 0 steps away from the current date then it would take the max_increment, then it will lose a 'slope' amount
    of increment for each unit of time defined in 'step'.
    Assume to have a 'date' column in the dataframe expressed in datetime format.
    """
    now = datetime.today()
    for hit in hits:
        date = df.iloc[hit['corpus_id']]['date']
        diff = (now - date).total_seconds()
        if step == 'year':
            diff = int(diff / (365*24*60*60))
        hit['score'] += max(0, max_increment - slope*diff)
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)
    return hits


def cluster_reranking(hits: list, df: pd.DataFrame, col_to_rank: str, num_clusters: int = 10) -> list:
    """
    This function first groups the search hits into clusters, according to their score and a uniform
    distribution of the "score space" into a fixed given number of clusters. Then each cluster is
    internally re-ranked (thus preserving the ordering of the clusters) according to metric value
    that is the column 'col_to_rank' in the given dataset.
    """
    # get minimum and maximum score
    min_score = min([hit['score'] for hit in hits])
    max_score = max([hit['score'] for hit in hits])
    # define score-width of each cluster
    cluster_width = (max_score - min_score) / num_clusters
    # define list of clusters with similar score
    clusters = []
    for i in range(num_clusters):
        clusters.append([hit for hit in hits if max_score-cluster_width*(i+1) <= hit['score'] <= max_score-cluster_width*i])
    # re-rank each cluster
    out = []
    for cluster in clusters:
        print(cluster)
        cluster_rankings = {hit['corpus_id']: df.iloc[hit['corpus_id']][col_to_rank] for hit in cluster}
        cluster_ordering = dict(sorted(cluster_rankings.items(), key=lambda x: x[1], reverse=True))
        print('rankings', cluster_ordering)
        for id in cluster_ordering.keys():
            hit_id = [hit for hit in cluster if hit['corpus_id'] == id][0]
            out.append(hit_id)
    assert len(out) == len(hits)
    return out



# ========== TRAINING ==========

def finetune_model(model, examples_ls: list, output_path: str, scores_ls: list = [], batch_size: int = 16):
    """
    Train an sBERT model using a given list of examples and return the best model found, according to a given evaluation metric  
    """
    # examples must be a list of tuples (sentence1, sentence2, label)
    train_examples = [InputExample(texts=[ex[0], ex[1]], label=ex[2]) for ex in examples_ls]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)  # shuffles data and creates batches
    
    # choose loss function
    train_loss = losses.CosineSimilarityLoss(model)
    # https://github.com/UKPLab/sentence-transformers/tree/master/sentence_transformers/losses
    
    # choose evaluation method
    if len(scores_ls) > 0:
        # scores must be a list of tuples (sentence1, sentence2, score)
        evaluator = evaluation.EmbeddingsSimilarityEvaluator([x[0] for x in scores_ls], [x[1] for x in scores_ls], [x[2] for x in scores_ls])
        # https://github.com/UKPLab/sentence-transformers/tree/master/sentence_transformers/evaluation
    
    model.fit(
        train_objectives=[  # more tuples can be passed to learn multiple tasks
            (train_dataloader, train_loss)  # trains on the given examples using the given loss function
        ],
        epochs=50,  # number of epochs for training
        # scheduler=scheduler,  # learning rate gradually increases from 0 to max 
        warmup_steps=100,  # after this many steps learning rate decreases from max to 0
        evaluator=evaluator,  # periodically evaluates performance on validation data, only best performing model is saved to disc
        evaluation_steps=500,  # period of the evaluation procedure
        output_path=output_path,  # storage
        save_best_model=True,  # stores best model according to evaluator
        checkpoint_path=output_path,  # folder to save checkpoints during training,
        checkpoint_save_steps=500,  # checkpoint period
        checkpoint_save_total_limit=0,  # maximum number of checkpoints to store
        show_progress_bar=True  # tqdm
    )
    
    return model



# ========== DATA ANALYSIS ==========

def text_length_histogram(text_ls: list, num_bins: int = 100):
    """
    Plot a frequency distribution histogram of the number of words in each element of a list of strings.
    """
    num_words = [len(text.split()) for text in text_ls]
    
    print('Maximum number of words:', max(num_words))
    print('Average number of words:', round(sum(num_words) / len(num_words)))
    
    plt.figure(figsize=(10, 5))
    plt.hist(num_words, bins=num_bins)
    plt.title('Distribution of the number of words in each text')
    plt.xlabel('number of words per text')
    plt.ylabel('frequency')
    plt.show()


def text_length_reduction(text: str, target_words: int, split_points: str = '.') -> list:
    """
    Splits a given around some characters and then re-aggregates the pieces
    so that each new block will have a length, in terms of number of words, 
    as close as possible to a given target value. 
    """
    # divide text in pieces
    pieces = text.split(split_points)
    # find groups of pieces so that number of words per group tries to match target
    groups = fair_sequence_division_order([len(piece.split()) for piece in pieces], target_words)
    # re-group pieces
    out = []
    idx = 0
    for group in groups:
        out.append(split_points.join(pieces[idx: idx+len(group)]).strip())
        idx += len(group)
    return out
