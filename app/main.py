import json
import gradio as gr

from searcher import Searcher


PROJECT_CONFIG = 'config.json'


config = json.load(open(PROJECT_CONFIG, encoding='utf-8'))
searcher = Searcher(config)


def user_search(query: str, limit: int, top_k: int, rank_by_date: bool, providers: list):
    return searcher.search(query=query, limit=int(limit), top_k=int(top_k), rank_by_date=rank_by_date, providers=providers)


demo = gr.Interface(
    fn=user_search,
    inputs=[
        gr.Textbox(lines=2, placeholder="Adventure in New York", label="User Query"),
        gr.Slider(minimum=1, maximum=128, value=10, step=1, label="Number of results to show"),
        gr.Slider(minimum=1, maximum=128, step=1, value=config['search']['top_k'], label="Number of results to rank"),
        gr.Checkbox(value=False, label="Prioritize latest contents"),
        gr.CheckboxGroup(choices=config['available_providers'])
    ],
    examples=config['tests'],
    outputs=[
        gr.JSON(label="Search Result")
    ],
    title="BertFlix",
    description="Enter a query to search a movie or TV show.\nDon't forget to filter the results according to your streaming services.",
    flagging_options=["Good", "Bad"]  # user feedback on result, automatically logged
)

demo.launch()
