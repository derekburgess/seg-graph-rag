import argparse
import os
import pandas as pd
from neo4j import GraphDatabase
import pyarrow.parquet as pq
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_utilization
from ragas.metrics.critique import correctness, coherence
from datasets import Dataset


def load_parquet(dataset):
    print(f"\nLoading dataset: {dataset}")
    if not dataset.endswith('.parquet'):
        dataset += '.parquet'
    file_path = os.path.join('./datasets', dataset)
    return pq.read_table(file_path).to_pandas()


def generate_query_embeddings(client, text):
    embedding_model = "text-embedding-3-small"
    response = client.embeddings.create(
        input=text,
        model=embedding_model
    )
    embeddings = response.data[0].embedding
    return embeddings


def find_relevant_chunks(client, df, query, num_chunks):
    print(f"Generating query embeddings...")
    query_embedding = generate_query_embeddings(client, query)
    print(f"Searching for the {num_chunks} most relevant chunks...")
    similarities = cosine_similarity([query_embedding], df['embedding'].tolist())
    top_indices = similarities.argsort()[0][-num_chunks:][::-1]
    relevant_chunks = df.iloc[top_indices]
    return relevant_chunks['chunk'].tolist()


def query_graph_for_chunks(driver, database, client, query, num_chunks):
    print(f"\nConnecting to database: {database}")
    def get_all_chunks(tx):
        result = tx.run("MATCH (t:TextChunk) RETURN t.text AS text, t.node_embedding AS embedding")
        chunks = [(record["text"], record["embedding"]) for record in result]
        return chunks

    with driver.session(database=database) as session:
        chunks = session.read_transaction(get_all_chunks)
    
    if not chunks:
        raise ValueError("No relevant TextChunk nodes found in the graph.")
    
    df = pd.DataFrame(chunks, columns=['chunk', 'embedding'])
    return find_relevant_chunks(client, df, query, num_chunks)


def generate_response(client, model, chunks, query):
    print(f"Sending query and chunks to: {model}", "\n")
    input_text = (
        f"Query: {query}\n\n"
        "Relevant Chunks:\n" + "\n".join(chunks) + "\n"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "This is a Retrieval-Augmented Generation (RAG) pipeline. You will receive relevant document chunks and are expected to return a response based on the query. Please structure your response as follows: 1) Restate the query, return a newline. 2) Provide a summary of the document chunks, return a newline. 3) Return a response to the query using the additional document chunks to inform your output. Ensure your response is clean, clear, and concise. DO NOT use markdown formatting in your response."
            },
            {"role": "user", "content": input_text}
        ]
    )
    return response.choices[0].message.content


def evaluate_with_ragas(query, response, chunks):
    ragas_data = Dataset.from_dict({
        "question": [query],
        "answer": [response],
        "contexts": [chunks]
    })
    results = evaluate(ragas_data, metrics=[faithfulness, answer_relevancy, context_utilization, correctness, coherence])
    return results


def list_datasets():
    datasets = [f for f in os.listdir('./datasets') if f.endswith('.parquet')]
    return [dataset[:-8] for dataset in datasets]


def main():
    parser = argparse.ArgumentParser(description='Search Parquet dataset or Neo4j graph for relevant chunks, pass chunks to OpenAI for response generation.')
    parser.add_argument('--dataset', type=str, help='Base name for the input Parquet dataset, no need to add .parquet extension.')
    parser.add_argument('--graph', action='store_true', help='Query the entire Neo4j graph for relevant chunks.')
    parser.add_argument('--database', type=str, help='Name of the Neo4j database to connect to.')
    parser.add_argument('--model', type=str, default="gpt-3.5-turbo", choices=["gpt-3.5-turbo", "gpt-4", "gpt-4o"], help='OpenAI model to use for response generation (options: gpt-3.5-turbo, gpt-4, gpt-4o).')
    parser.add_argument('--query', type=str, help='Query, also used to find and return relevant chunks in the dataset or graph.')
    parser.add_argument('--chunks', type=int, default=50, help='Number of relevant chunks to return.')
    parser.add_argument('--list', action='store_true', help='Returns a list of available datasets.')

    args = parser.parse_args()
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    database = args.database
    driver = GraphDatabase.driver(uri, auth=(username, password))
    client = OpenAI()

    if args.list:
        datasets = list_datasets()
        print("\nAvailable datasets:")
        for dataset in datasets:
            print(f"- {dataset}")
        print("\n")
    elif (args.dataset or args.graph) and args.query:
        if args.dataset:
            df = load_parquet(args.dataset)
            relevant_chunks = find_relevant_chunks(client, df, args.query, args.chunks)
        else:
            relevant_chunks = query_graph_for_chunks(driver, database, client, args.query, args.chunks)
        
        if relevant_chunks:
            response = generate_response(client, args.model, relevant_chunks, args.query)
            print(f"Response:\n{response}\n")
        
            ragas_metrics = evaluate_with_ragas(args.query, response, relevant_chunks)
            print("\nRagas Metrics:")
            print(f"Faithfulness: {ragas_metrics['faithfulness']}")
            print(f"Answer Relevance: {ragas_metrics['answer_relevancy']}")
            print(f"Context Utilization: {ragas_metrics['context_utilization']}")
            print(f"Correctness: {ragas_metrics['correctness']}")
            print(f"Coherence: {ragas_metrics['coherence']}", "\n")
        else:
            print("No relevant chunks found. Adjust your query or select a different dataset/graph.")
    else:
        parser.print_help()

    driver.close()

if __name__ == "__main__":
    main()