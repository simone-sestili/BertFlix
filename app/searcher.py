import os
import json
import pandas as pd

from datetime import datetime
from justwatch import JustWatch

from get_data import justwatch_dataset
from utils.utils import list_intersection
from utils.sbert import load_data, load_model, load_embeddings, text_search, text_rerank, date_reranking, text_length_histogram, processing, cluster_reranking


class Searcher:
    
    def __init__(self, config: dict):
        
        self.config = config
        
        self.justwatch = JustWatch(country=self.config['dataset']['country'])
        
        # creates data folder
        if not os.path.exists(self.config['data_folder']):
            os.makedirs(self.config['data_folder'], exist_ok=True)
    
        self.bi_encoder = load_model(self.config['model']['folder'], self.config['model']['bi_encoder'])
        if config['model']['cross_encoder']:
            self.cross_encoder = load_model(self.config['model']['folder'], self.config['model']['cross_encoder'])
        
        if self.config['update_dataset'] or (self.config['dataset']['filename'] not in os.listdir(self.config['data_folder'])):
            # download and pre-process dataset
            self.dataset = justwatch_dataset(self.justwatch, self.config)
            self.dataset['text'] = self.dataset['text'].apply(lambda x: processing(x, target_words=self.config['dataset']['target_words']))
            self.dataset.to_csv(os.path.join(self.config['data_folder'], self.config['dataset']['filename']), index=False)
        else:
            # load local dataset
            self.dataset = load_data(self.config['data_folder'], self.config['dataset']['filename'])
        
        if self.config['update_embeddings'] or (self.config['embeddings']['filename'] not in os.listdir(self.config['data_folder'])):
            # compute embeddings
            self.corpus_names, self.corpus_embeddings = load_embeddings(
                data=self.dataset['text'].tolist(),
                model=self.bi_encoder,
                embeddings_path=os.path.join(config['data_folder'], config['embeddings']['filename']),
                batch_size=config['embeddings']['batch_size']
            )
        else:
            # load local embeddings
            self.corpus_names, self.corpus_embeddings = load_embeddings(
                data=self.dataset['text'].tolist(),
                model=self.bi_encoder,
                embeddings_path=os.path.join(config['data_folder'], config['embeddings']['filename']),
                use_precomputed=True
            )
    
    
    def show_data_distribution(self) -> None:
        text_length_histogram(self.dataset['text'].tolist())
    
    
    def filter_by_providers(self, providers_ls: list):
        """
        This function filters the dataset, as well as the corresponding embeddings, by keeping
        only the titles for which at least one provider is in the given list.
        """
        if len(providers_ls) == 0:
            # no filters
            return self.dataset, self.corpus_names, self.corpus_embeddings
        else:
            # append to sub_dataset only the rows which share at least a provider with providers_ls
            sub_dataset = []
            sub_corpus_names = []
            sub_corpus_embeddings = []
            for i, el in enumerate(self.dataset.to_dict('records')):
                el_providers = el['providers'].split('[')[1].split(']')[0]
                el_providers = [p.strip().strip('\'') for p in el_providers.split(',')]
                if len(list_intersection(el_providers, providers_ls)) > 0:
                    sub_dataset.append(el)
                    # keep only corpus names / embeddings which appear in the filtered_dataset, with the same index
                    sub_corpus_names.append(self.corpus_names[i])
                    sub_corpus_embeddings.append(self.corpus_embeddings[i])
            sub_dataset = pd.DataFrame(sub_dataset)
            return sub_dataset, sub_corpus_names, sub_corpus_embeddings        

    
    def search(self, query: str, limit: int = 10, top_k: int = 32, rank_by_date: bool = False, providers: list = []) -> list:
        
        # consider only the sub-dataset of titles related to given providers
        self.sub_dataset, self.sub_corpus_names, self.sub_corpus_embeddings = self.filter_by_providers(providers)
        
        # perform bi-encoder search
        search_hits = text_search(
            query=query,
            df=self.sub_dataset,
            corpus_embeddings=self.sub_corpus_embeddings,
            model=self.bi_encoder,
            top_k=top_k,
            results_to_show=0
        )
        
        # perform re-ranking on subset of results using cross-encoder
        if self.config['model']['cross_encoder']:
            search_hits = text_rerank(
                query=query,
                df=self.sub_dataset,
                corpus_names=self.sub_corpus_names,
                search_hits=search_hits,
                model=self.cross_encoder,
                results_to_show=0
            )
        
        # re-rank result by giving more importance to the most recent results
        if rank_by_date:
            self.sub_dataset['date'] = self.sub_dataset['year'].apply(lambda y: datetime.strptime(str(y), '%Y'))
            search_hits = date_reranking(
                hits=search_hits, 
                df=self.sub_dataset, 
                max_increment=self.config['search']['date_reranking']['max_increment'], 
                slope=self.config['search']['date_reranking']['slope'],
                step=self.config['search']['date_reranking']['step']
            )
        
        # re-rank results with similar scores according to their 'rating' value
        if len(search_hits) > 1:
            search_hits = cluster_reranking(
                hits=search_hits,
                df=self.sub_dataset,
                col_to_rank='rating',
                num_clusters=self.config['search']['rating_reranking']['num_clusters']
            )
        
        # get rows of sub_dataset corresponding to results
        result_ids = [el['corpus_id'] for el in search_hits]
        results = self.sub_dataset.iloc[result_ids][['title', 'year', 'rating', 'providers', 'text']].to_dict('records')
        
        return results[:limit]



if __name__ == '__main__':
    PROJECT_CONFIG = 'config.json'
    config = json.load(open(PROJECT_CONFIG, encoding='utf-8'))
    searcher = Searcher(config)
    # searcher.show_data_distribution()
    print(searcher.search(query='New York', rank_by_date=True, providers=['Netflix']))
