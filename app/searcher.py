import os
import json

from datetime import datetime
from justwatch import JustWatch

from get_data import justwatch_dataset
from utils.utils import list_intersection
from utils.sbert import load_data, load_model, load_embeddings, text_search, text_rerank, date_reranking, text_length_histogram, processing


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
        
        if self.config['dataset']['filename'] not in os.listdir(self.config['data_folder']):
            # download and pre-process dataset
            self.dataset = justwatch_dataset(self.justwatch, self.config)
            self.dataset['text'] = self.dataset['text'].apply(lambda x: processing(x, target_words=self.config['dataset']['target_words']))
            self.dataset.to_csv(os.path.join(self.config['data_folder'], self.config['dataset']['filename']), index=False)
        else:
            # load local dataset
            self.dataset = load_data(self.config['data_folder'], self.config['dataset']['filename'])
        
        if self.config['compute_embeddings'] or (self.config['embeddings']['filename'] not in os.listdir(self.config['data_folder'])):
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
    
    
    def merge_duplicates(self, res: list) -> list:
        """
        There may be some scenarios in which two different items have the same title and year,
        but different ids since they have different pages for different providers; this function merges them.
        """
        out = []
        unique_els = []
        for el in res:
            if f"{el['title']}-{el['year']}" not in unique_els:
                unique_els.append(f"{el['title']}-{el['year']}")
                group = [x for x in res if x['title'] == el['title'] and x['year'] == el['year']]
                group_ratings = [x['rating'] for x in group]
                group_providers = []
                for x in group:
                    group_providers.extend(x['providers'])
                out.append({
                    'title': group[0]['title'],
                    'year': group[0]['year'],
                    'text': group[0]['text'],
                    'rating': round(sum(group_ratings) / len(group_ratings), 1),
                    'providers': group_providers
                })
        return out
    
    
    def filter_by_providers(self, res: list, providers_ls: list) -> list:

        if len(providers_ls) == 0:
            return res
        
        out = []
        for el in res:
            if len(list_intersection(el['providers'], providers_ls)) > 0:
                out.append(el)
        return out

    
    def search(self, query: str, limit: int = 10, top_k: int = 32, rank_by_date: bool = False, providers: list = []) -> list:
        
        search_hits = text_search(
            query=query,
            df=self.dataset,
            corpus_embeddings=self.corpus_embeddings,
            model=self.bi_encoder,
            top_k=top_k,
            results_to_show=0
        )
        
        if self.config['model']['cross_encoder']:
            search_hits = text_rerank(
                query=query,
                df=self.dataset,
                corpus_names=self.corpus_names,
                search_hits=search_hits,
                model=self.cross_encoder,
                results_to_show=0
            )
        
        if rank_by_date:
            self.dataset['date'] = self.dataset['year'].apply(lambda y: datetime.strptime(str(y), '%Y'))
            search_hits = date_reranking(
                hits=search_hits, 
                df=self.dataset, 
                max_increment=self.config['search']['date_reranking']['max_increment'], 
                slope=self.config['search']['date_reranking']['slope'],
                step=self.config['search']['date_reranking']['step']
            )
        
        # get sub-dataset of results
        result_ids = [el['corpus_id'] for el in search_hits]
        results = self.dataset.iloc[result_ids].to_dict('records')

        results = self.merge_duplicates(results)
        
        return self.filter_by_providers(results, providers)[:limit]



if __name__ == '__main__':
    PROJECT_CONFIG = 'config.json'
    config = json.load(open(PROJECT_CONFIG, encoding='utf-8'))
    searcher = Searcher(config)
    # searcher.show_data_distribution()
    print(searcher.search(query='New York', rank_by_date=True, providers=['Netflix']))
