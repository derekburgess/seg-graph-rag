# SEG-GRAPH-RAG

Segmented or Graph Retrieval Augmented Generation. At its core, this is a simple 2 command RAG CLI. It can create individual Parquet files per document (SEG), and provide document querying using OpenAI (RAG). It also works with Neo4j and provides commands and options to process parquet datasets into a graph database, allowing for graph based RAG.

Models: `text-embedding-3-small` and `gpt-3.5-turbo`, `gpt-4,` or `gpt-4o`.


## Setup

Recommended:

Create an environment, such as:

`conda create --name SEG-RAG python=3.12`


Required:

Configure the environment and install the Python project.

`conda env config vars set OPENAI_API_KEY=value`

`cd SEG-RAG/`

`pip install -r requirements.txt`

`pip install .`

Everything is hardcoded to use a local datasets directory:

`mkdir datasets`


Optional:

If you want to use the graph approach, download and install the Neo4j desktop app.

Configure the environment to work with Neo4j:

`conda env config vars set ENV_VAR=value`

`NEO4J_URI` (bolt://localhost:7687)

`NEO4J_USERNAME` (neo4j)

`NEO4J_PASSWORD` (you set)

Create a Neo4j database called seg-graph-rag.


## Usage

To process a PDF into vectors, stored as a Parquet file, run:

`seg --document PATH --dataset NAME`

e.x., `seg --document Documents/test_documents/test.pdf --dataset test_dataset`

Note: There is no need to add .parquet when creating a dataset. You can also continue to add PDF's to a dataset by calling it again.


To process Parquet datasets into the graph database:

`graph --dataset NAME`

e.x., `graph --dataset test_dataset`


To querty and existing parquert dataset, run:

`rag --dataset NAME --model MODEL --chunks INT --query QUERY`

e.x., `rag --dataset test_dataset --model gpt-4 --chunks 100 --query "What is this about?"`


You can pass an argument to query the entire graph database like so:

`rag --graph --model MODEL --chunks INT --query QUERY`


You can list available datasets by running:

`rag --list`


You can also spin up a Gradio based web UI for iteracting with the RAG side by running:

`rag --interface`

