# seg-graph-rag

Segmented and/or Graph Retrieval Augmented Generation. A simple 2 or 3 command RAG CLI. Create Parquet datasets from PDF documents (SEG), process datasets into Neo4j (GRAPH), and query datasets or graphs using OpenAI (RAG).

Models: `text-embedding-3-small` and `gpt-3.5-turbo`, `gpt-4,` or `gpt-4o`.


## Setup

Create an environment, such as:

`conda create --name seg-graph-rag python=3.12`

Configure the environment and install the Python project:

`conda env config vars set OPENAI_API_KEY=value`

`cd SEG-RAG/`

`pip install -r requirements.txt`

`pip install .`

Everything is hardcoded to use a local datasets directory:

`mkdir datasets`

At this point you can operate the Segmented RAG using only 2 commands.

If you want to use the graph approach, download and install the Neo4j desktop app, which will handle the driver install and provide some useful graph GUI tools.

After installing Neo4j GUI, create a Project and DBMS, after which you can install the needed Graph Design Science Library plugin.

Configure the environment to work with Neo4j:

`conda env config vars set ENV_VAR=value`

`NEO4J_URI` (bolt://localhost:7687)

`NEO4J_USERNAME` (neo4j)

`NEO4J_PASSWORD` (you set)

At this point you will need to use all 3 commands to create Parquet files and process them into the Neo4j graph.


## Usage

To process a PDF into vectors, stored as a Parquet dataset run:

`seg --document PATH --dataset NAME --images BINARY`

e.x., `seg --document Documents/test_documents/test.pdf --dataset test_dataset`

There is no need to add .parquet when creating a dataset. You can also continue to add PDF's to a dataset by calling it again with a different PDF.

Adding `--images` will send the images in the PDF to OpenAI for summary and store the summaries as chunks.

To process Parquet datasets into the graph database:

`graph --dataset NAME --database NAME`

e.x., `graph --dataset test_dataset --database neo4j`

To RAG a Parquert dataset run:

`rag --dataset NAME --model MODEL --chunks INT --query QUERY`

e.x., `rag --dataset test_dataset --model gpt-4 --chunks 50 --query "What is this about?"`

To RAG the graph database run:

`rag --graph --database NAME --model MODEL --chunks INT --query QUERY`

e.x., `rag --graph --database neo4j --model gpt-4 --chunks 50 --query "What is this about?"`

You can list available datasets by running:

`rag --list`

