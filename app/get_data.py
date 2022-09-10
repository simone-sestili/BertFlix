import os
import json
import time
import pandas as pd

from tqdm import tqdm
from requests.exceptions import HTTPError
from justwatch import JustWatch



def get_providers_names(jw: JustWatch) -> list:
    return [el['clear_name'] for el in jw.get_providers()]


def get_providers_map(jw: JustWatch) -> dict:
    res = {}
    for el in jw.get_providers():
        res[el['id']] = el['clear_name']
    return res


def get_all_title_ids(jw: JustWatch) -> list:
    """
    Search in all the pages of the JustWatch dataset to retrieve the list of all the IDs
    """
    # get number of pages
    num_pages = jw.search_for_item()['total_pages']
    res = []
    for page in range(1, num_pages+1):
        res.extend([el['id'] for el in jw.search_for_item(page=page)['items']])
    return res


def get_jw_title(jw: JustWatch, title_id: str):
    """
    Performs a request to the JustWatch API in order to retrieve the
    information about a title given its numerical id.
    """
    try:
        # assume it is a movie
        info = jw.get_title(title_id=title_id, content_type='movie')
    except HTTPError as e:
        if e.response.status_code == 429:
            # too many requests, wait one second and retry
            time.sleep(1)
            info = get_jw_title(jw, title_id)
        elif e.response.status_code == 404:
            try:
                # it is not a movie -> assume it is a tv show
                info = jw.get_title(title_id=title_id, content_type='show')
            except HTTPError as e:
                if e.response.status_code == 429:
                    # too many requests, wait one second and retry
                    time.sleep(1)
                    info = get_jw_title(jw, title_id)
                else:
                    print(f'Troubles_1 retrieving {title_id}')
                    print('Exception:', e)
                    print('Exception status code', e.response.status_code)
        else:
            print(f'Troubles_2 retrieving {title_id}')
            print('Exception:', e)
            print('Exception status code', e.response.status_code)
    return info


def merge_duplicates(res: list) -> list:
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
            group_ratings = [x['rating'] for x in group if x['rating'] != '']
            group_providers = []
            for x in group:
                group_providers.extend(x['providers'])
            out.append({
                'title': group[0]['title'],
                'year': group[0]['year'],
                'text': group[0]['text'],
                'rating': round(sum(group_ratings) / len(group_ratings), 1) if len(group_ratings) > 0 else '',
                'providers': group_providers
            })
    return out


def justwatch_dataset(jw: JustWatch, config: dict) -> pd.DataFrame:
    """
    This function creates a dataset of movies and tv shows listed on the
    JustWatch platform, it retrieves all the information about the titles
    and creates a pandas DataFrame with the useful information only.
    """
    # get list of all title ids     
    all_title_ids = get_all_title_ids(jw)
    all_title_ids = list(set(all_title_ids))
    # get map of ids to providers
    providers_map = get_providers_map(jw)
    
    # create dataset by adding title's metadata
    out = []
    for title_id in tqdm(all_title_ids):
        info = get_jw_title(jw, title_id)
        if 'title' in info.keys() and 'original_release_year' in info.keys() and 'short_description' in info.keys():
            
            # select only titles release after a given minimum year
            year = info['original_release_year']
            if year >= config['dataset']['min_year']:
                tmp = {
                    'title_id': title_id,
                    'title': info['title'],
                    'year': year,
                    'text': info['short_description']
                }
                
                # add average rating
                tmp['rating'] = ''
                if 'scoring' in info.keys():
                    ratings = []
                    for score in info['scoring']:
                        if score['provider_type'].split(':')[1] == 'score':
                            ratings.append(score['value'])
                    if len(ratings) > 0:
                        tmp['rating'] = round(sum(ratings) / len(ratings), 1)
                
                # add list of providers
                tmp['providers'] = []
                if 'offers' in info.keys():
                    providers = []
                    for offer in info['offers']:
                        if offer['monetization_type'] in config['search']['monetization']:
                            if offer['provider_id'] in providers_map.keys():
                                provider_name = providers_map[offer['provider_id']]
                                if provider_name not in providers:
                                    providers.append(provider_name)
                    if len(providers) > 0:
                        tmp['providers'] = providers
                
                out.append(tmp)
    
    # merge duplicates by different providers
    out = merge_duplicates(out)

    # convert to dataframe and remove duplicates (i.e. same title, same plot, same year)
    df_out = pd.DataFrame(out)
    print(f'Retrieved {len(df_out)} titles')
    df_out_2 = df_out.drop_duplicates(subset=['title', 'year', 'text'], keep='first')
    print('Removed', len(df_out)-len(df_out_2))
    return df_out



if __name__ == '__main__':

    config = json.load(open('config.json'))
    jw = JustWatch(country=config['dataset']['country'])
    print(get_providers_names(jw))
    print(get_providers_map(jw))
    dataset = justwatch_dataset(jw, config)
    dataset.to_csv(os.path.join(config['data_folder'], config['dataset']['filename']))
