# BertFlix

Have you ever tried to search for a content on Netflix? Well, good luck! In 2022 the integrated search is still basically a syntactic search, which returns only the contents whose title matches your query, and then it goes almost on random.

The purpose of this application is to implement a semantic search engine on a database of movies and shows from several platforms, such an engine is therefore able to search content that matches the *meaning* of your query, and not the exact words that your used, and for doing that it searches on the whole plot.

The application also allows to filter out content from specific platforms that you do not use, to influence the search engine to prioritize newer title (all other things being equal) and it is also designed to automatically give priority to titles with higher average scores.

Please consider that the platform distribution of a title is highly country-dependant, therefore this application is designed to work for an *italian audience only*.

# Installation
## Using Python

The application is entirely built in Python 3.9, all the necessary dependencies can be installed by running:
```
pip install -r requirements.txt
```
When doing so it is suggested to create an ad-hoc Python environment with all the necessary dependencies. Then the application can be executed by simply running:
```
python main.py
```
## Using Docker
The application can be entirely deployed as a ready-for Docker container. The Docker image can be created by simply running the script `helper_make.sh` and then the Docker container can be run by simply running the script `run.sh`.

# Back-end

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
