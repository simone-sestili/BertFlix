# BerTeX

The purpose of this application is to implement a semantic search on a database of documents written in LaTeX.

# Installation
## Using Python

The application is entirely built in Python 3.9, all the necessary dependencies can be installed by running:
```
pip install -r requirements.txt
```
If Anaconda is already installed, a clean installation of the application can be done by simply running `helper_make.bat`, which creates an ad-hoc environment with all the necessary dependencies. In this scenario, the application can be executed by simply running `run.bat`.

## Back-end

The semantic search is powered by a BERT model, using the [sentence-transformers](https://sbert.net/) public library; in particular, the search is done by means of a bi-encoder, to filter out the best $K$ results, and a cross-encoder to improve results ranking; both this models are defined in the application's configuration file at `/app/config.json`.

The whole application will be executed by running `python main.py`, since this will handle the data retrieval, the models download, the embeddings pre-computing, the actual search process, and the front-end implementation.

# Front-end
The application is accessed through the `gradio` tool for developing web interfaces (more information are available at [gradio.app](https://gradio.app)). The search engine is accessed by using 3 input parameters:

- `User query`: free text describing what the user is searching for.
- `Number of results to show` [Optional]: number of results shown in the web interface.
- `Number of results to rank` [Optional]: number of results extracted from the bi-encoder search and used to compute re-ranking through cross-encoder; note that a higher number will potentially increase output quality, but it will also increase computing time.

The output is displayed as a JSON file, in which each element is defined by the document's name, the corresponding section title, and the actual text that matched the user query. 

# Dataset
The dataset of titles is updated manually by scraping the [JustWatch](https://www.justwatch.com/) platform, a popular website with information about movies/tv-shows, such as plots, ratings and most importantly the platforms on which the items are currently available. The website's contents are accessed through an unofficial [Python wrapper](https://github.com/dawoudt/JustWatchAPI) of the JustWatch API, which is loaded in the project as a library from PyPI.


cluster similar cross-scores
re-rank clusters by user evaluation

show images
