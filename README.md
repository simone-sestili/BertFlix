# BerTeX

The purpose of this application is to implement a semantic search on a database of documents written in LaTeX.

# Installation
## Using Python

The application is entirely build in Python 3.9, all the necessary dependencies can be installed by running:
```
pip install -r requirements.txt
```
When doing so it is suggested to create an ad-hoc Python environment with all the necessary dependencies. Then it is required to use the following commands, in order, to create the dataset, create the model embeddings, and finally run the application:
```
python main.py
```
## Using Docker
The application can be entirely deployed as a ready-for Docker container. The Docker image can be created by simply running the script `helper_make.sh` and then the Docker container can be run by simply running the script `run.sh`.

# Back-end
The semantic search is powered by a BERT model, using the `sentence-transformers` public library; in particular, the search is done by means of a bi-encoder, to filter out the best $K$ results, and a cross-encoder to improve results ranking; both this models are defined in the application's configuration file at `/app/config.json`.

The whole application will be executed by running `python main.py`, since this will handle the data retrieval, the models download, the embeddings pre-computing, the actual search process, and the front-end implementation.

# Front-end
The application is accessed through the `gradio` tool for developing web interfaces (more information are available at [gradio.app](https://gradio.app)). The search engine is accessed by using 3 input parameters:

- `User query`: free text describing what the user is searching for.
- `Number of results to show` [Optional]: number of results shown in the web interface.
- `Number of results to rank` [Optional]: number of results extracted from the bi-encoder search and used to compute re-ranking through cross-encoder; note that a higher number will potentially increase output quality, but it will also increase computing time.

The output is displayed as a JSON file, in which each element is defined by the document's name, the corresponding section title, and the actual text that matched the user query. 



cluster similar cross-scores
re-rank clusters by user evaluation

show images
