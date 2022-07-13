import os
import json

import pandas as pd

from sbert import text_length_histogram, text_length_reduction


def processing(text: str, target_words: int) -> str:
    # split text to reduce size
    # sub_paragraphs = text_length_reduction(text, target_words, '\n')
    return text.lower().strip('\n').strip()



if __name__ == '__main__':
    
    PROJECT_CONFIG = 'config.json'
    config = json.load(open(PROJECT_CONFIG, encoding='utf-8'))
    
    data = pd.read_csv(os.path.join(config['data_folder'], config['dataset']['filename']))
    data['text'] = data['text'].apply(lambda x: processing(x, target_words=config['dataset']['target_words']))
    data.to_csv(os.path.join(config['data_folder'], config['dataset']['filename']))
        
    text_length_histogram(data['text'].tolist())
