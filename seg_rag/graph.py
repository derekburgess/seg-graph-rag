import os
import argparse
import pandas as pd
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity


uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
database = "seg-graph-rag"
driver = GraphDatabase.driver(uri, auth=(username, password))


def execute_query(query, parameters=None):
    with driver.session(database=database) as session:
        session.run(query, parameters)


def create_dataset_node(dataset_name):
    query = """
    MERGE (d:Dataset {name: $name})
    """
    parameters = {"name": dataset_name}
    execute_query(query, parameters)


def node_exists(text, dataset_name):
    query = """
    MATCH (t:TextChunk {text: $text, dataset: $dataset})
    RETURN t
    """
    parameters = {
        "text": text,
        "dataset": dataset_name
    }
    with driver.session(database=database) as session:
        result = session.run(query, parameters)
        return result.single() is not None


def create_nodes(df, dataset_name):
    create_dataset_node(dataset_name)
    
    for index, row in df.iterrows():
        unique_id = f"{dataset_name}_{index}"
        if not node_exists(row['chunk'], dataset_name):
            query = """
            MATCH (d:Dataset {name: $dataset})
            CREATE (t:TextChunk {id: $id, text: $text, embedding: $embedding, dataset: $dataset})
            CREATE (d)-[:CONTAINS]->(t)
            """
            parameters = {
                "id": unique_id,
                "text": row['chunk'],
                "embedding": row['embedding'],
                "dataset": dataset_name
            }
            execute_query(query, parameters)


def relationship_exists(id1, id2):
    query = """
    MATCH (a:TextChunk {id: $id1})-[r:SIMILAR_TO]->(b:TextChunk {id: $id2})
    RETURN r
    """
    parameters = {"id1": id1, "id2": id2}
    with driver.session(database=database) as session:
        result = session.run(query, parameters)
        return result.single() is not None


def create_relationships(df, dataset_name):
    embeddings = df['embedding'].tolist()
    similarity_matrix = cosine_similarity(embeddings)
    
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            similarity_score = similarity_matrix[i][j]
            if similarity_score > 0.8:
                id1 = f"{dataset_name}_{i}"
                id2 = f"{dataset_name}_{j}"
                if not relationship_exists(id1, id2):
                    query = """
                    MATCH (a:TextChunk {id: $id1}), (b:TextChunk {id: $id2})
                    CREATE (a)-[:SIMILAR_TO {score: $score}]->(b)
                    """
                    parameters = {
                        "id1": id1,
                        "id2": id2,
                        "score": similarity_score
                    }
                    execute_query(query, parameters)


def process_dataset(dataset_path, dataset_name):
    print(f"\nProcessing dataset: {dataset_name}...", "\n")
    df = pd.read_parquet(dataset_path)
    create_nodes(df, dataset_name)
    create_relationships(df, dataset_name)


def main():
    parser = argparse.ArgumentParser(description='Populate Neo4j graph with data from a specific Parquet dataset.')
    parser.add_argument('--dataset', type=str, required=True, help='Base name for the output Parquet dataset, no need to add .parquet extension.')

    args = parser.parse_args()

    dataset_name = args.dataset if args.dataset.endswith('.parquet') else f'{args.dataset}.parquet'
    dataset_path = f'./datasets/{dataset_name}'

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"The dataset: {dataset_path} does not exist. Please check your path and try again.")

    process_dataset(dataset_path, dataset_name)

    driver.close()

if __name__ == "__main__":
    main()