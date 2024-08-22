import argparse
import os
import pandas as pd
from neo4j import GraphDatabase
import pyarrow.parquet as pq
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import gradio as gr


def load_parquet(dataset):
    print(f"\nLoading dataset: {dataset}", "\n")
    if not dataset.endswith('.parquet'):
        dataset += '.parquet'
    file_path = os.path.join('./datasets', dataset)
    return pq.read_table(file_path).to_pandas()


def generate_embeddings(client, text):
    print(f"Generating query embeddings...", "\n")
    embedding_model = "text-embedding-3-small"
    response = client.embeddings.create(
        input=text,
        model=embedding_model
    )
    embeddings = response.data[0].embedding
    return embeddings


def find_relevant_chunks(client, df, query, num_chunks):
    query_embedding = generate_embeddings(client, query)
    print(f"Using consine similarity to return the {num_chunks} most relevant chunks...", "\n")
    similarities = cosine_similarity([query_embedding], df['embedding'].tolist())
    top_indices = similarities.argsort()[0][-num_chunks:][::-1]
    relevant_chunks = df.iloc[top_indices]
    return relevant_chunks['chunk'].tolist()


def query_graph_for_chunks(driver, client, query, num_chunks):
    print(f"\nQuerying Neo4j graph for chunks...", "\n")
    def get_all_chunks(tx):
        result = tx.run("MATCH (t:TextChunk) RETURN t.text AS text, t.embedding AS embedding")
        chunks = [(record["text"], record["embedding"]) for record in result]
        return chunks

    with driver.session() as session:
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
            {"role": "system", "content": "This is a Retrieval-Augmented Generation (RAG) pipeline. You will receive relevant document chunks and are expected to return a response based on the query. Please structure your response as follows: 1) Restate the query, return a newline. 2) Provide a summary of the document chunks, return a newline. 3) Return a response to the query using the additional document chunks to inform your output. Ensure your response is clean, clear, and concise."
            },
            {"role": "user", "content": input_text}
        ]
    )
    return response.choices[0].message.content


def list_datasets():
    datasets = [f for f in os.listdir('./datasets') if f.endswith('.parquet')]
    return [dataset[:-8] for dataset in datasets]


def gradio_interface(client, driver):
    def process_query(use_graph, dataset, model, query, chunks):
        if use_graph:
            relevant_chunks = query_graph_for_chunks(driver, client, query, chunks)  # Correct order: driver first
        else:
            df = load_parquet(dataset)
            relevant_chunks = find_relevant_chunks(client, df, query, chunks)
        
        if relevant_chunks:
            response = generate_response(client, model, relevant_chunks, query)
            print(f"Response:\n{response}\n")
            return response
        else:
            return "No relevant chunks found, adjust your query or select a different dataset."

    datasets = list_datasets()

    with gr.Blocks() as interface:
        with gr.Row():
            use_graph = gr.Checkbox(label="Use Neo4j Graph")
            dataset = gr.Dropdown(choices=datasets, label="Select a Dataset")
            model = gr.Dropdown(choices=["gpt-3.5-turbo", "gpt-4", "gpt-4o"], label="Select a Model")
            chunks = gr.Slider(minimum=1, maximum=100, value=20, step=1, label="Number of Chunks")

        with gr.Row():
            query = gr.Textbox(label="Enter a Query")

        output = gr.Textbox(label="Response")

        submit_btn = gr.Button("Submit")
        submit_btn.click(fn=process_query, inputs=[use_graph, dataset, model, query, chunks], outputs=output)

    interface.launch(share=True)


def main():
    parser = argparse.ArgumentParser(description='Search Parquet dataset or Neo4j graph for relevant chunks, pass chunks to OpenAI for response generation.')
    parser.add_argument('--dataset', type=str, help='Base name for the output Parquet dataset, no need to add .parquet extension.')
    parser.add_argument('--graph', action='store_true', help='Query the entire Neo4j graph for relevant chunks.')
    parser.add_argument('--model', type=str, default="gpt-3.5-turbo", choices=["gpt-3.5-turbo", "gpt-4", "gpt-4o"], help='OpenAI model to use for response generation (options: gpt-3.5-turbo, gpt-4, gpt-4o).')
    parser.add_argument('--query', type=str, help='Query, also used to find and return relevant chunks in the dataset or graph.')
    parser.add_argument('--chunks', type=int, default=50, help='Number of relevant chunks to return.')
    parser.add_argument('--list', action='store_true', help='Returns a list of available datasets.')
    parser.add_argument('--interface', action='store_true', help='Launch Gradio interface, sharing set to true.')

    args = parser.parse_args()
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    database = "seg-graph-rag"
    driver = GraphDatabase.driver(uri, auth=(username, password), database=database)
    client = OpenAI()

    if args.interface:
        gradio_interface(client, driver)
    elif args.list:
        datasets = list_datasets()
        print("\nAvailable datasets:")
        for dataset in datasets:
            print(f"- {dataset}")
        print("\n")
    elif args.dataset and args.query:
        df = load_parquet(args.dataset)
        relevant_chunks = find_relevant_chunks(client, df, args.query, args.chunks)
        if relevant_chunks:
            response = generate_response(client, args.model, relevant_chunks, args.query)
            print(f"{response}", "\n")
        else:
            print("No relevant chunks found, adjust your query or select a different dataset.")
    elif args.graph and args.query:
        relevant_chunks = query_graph_for_chunks(driver, client, args.query, args.chunks)
        if relevant_chunks:
            response = generate_response(client, args.model, relevant_chunks, args.query)
            print(f"{response}", "\n")
        else:
            print("No relevant chunks found in the graph.")
    else:
        parser.print_help()

    driver.close()

if __name__ == "__main__":
    main()