import os
import json
import pandas as pd

from justwatch import JustWatch



def get_providers_names(jw: JustWatch) -> list:
    return [el['clear_name'] for el in jw.get_providers()]


def get_providers_map(jw: JustWatch) -> dict:
    res = {}
    for el in jw.get_providers():
        res[el['id']] = el['clear_name']
    return res


def get_all_title_ids(jw: JustWatch):
    """
    Search in all the pages of the JustWatch dataset to retrieve the list of all the IDs
    """
    # get number of pages
    num_pages = jw.search_for_item()['total_pages']
    res = []
    for page in range(1, num_pages+1):
        res.extend([el['id'] for el in jw.search_for_item(page=page)['items']])
    return res


def justwatch_dataset(jw: JustWatch, config: dict):
    # get list of all title ids
    all_title_ids = get_all_title_ids(jw)
    # create dataset by adding title's metadata
    out = []
    for title_id in all_title_ids:
        try:
            info = jw.get_title(title_id=title_id)
        except:
            print(f'Troubles retrieving {title_id}')
        if 'title' in info.keys() and 'original_release_year' in info.keys() and 'short_description' in info.keys():
            year = info['original_release_year']
            if year >= config['dataset']['min_year']:
                out.append({
                    'title_id': title_id,
                    'title': info['title'],
                    'year': year,
                    'text': info['short_description']
                })
    print(f'Retrieved {len(out)} titles')
    return pd.DataFrame(out)



if __name__ == '__main__':

    config = json.load(open('config.json'))
    jw = JustWatch(country=config['dataset']['country'])
    print(get_providers_names(jw))
    print(get_providers_map(jw))
    dataset = justwatch_dataset(jw, config)
    dataset.to_csv(os.path.join(config['data_folder'], config['dataset']['filename']))
