import os
import argparse
import pandas as pd
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity


def execute_query(driver, database, query, parameters=None):
    with driver.session(database=database) as session:
        session.run(query, parameters)


def create_dataset_node(driver, database, dataset_name):
    query = """
    MERGE (d:Dataset {name: $name})
    """
    parameters = {"name": dataset_name}
    execute_query(driver, database, query, parameters)


def node_exists(driver, database, text, dataset_name):
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


def create_nodes(driver, database, df, dataset_name):
    create_dataset_node(driver, database, dataset_name)
    
    for index, row in df.iterrows():
        unique_id = f"{dataset_name}_{index}"
        if not node_exists(driver, database, row['chunk'], dataset_name):
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
            execute_query(driver, database, query, parameters)


def get_all_text_chunks(driver, database):
    query = """
    MATCH (t:TextChunk)
    RETURN t.id AS id, t.embedding AS embedding
    """
    with driver.session(database=database) as session:
        result = session.run(query)
        chunks = [(record["id"], record["embedding"]) for record in result]
    return chunks


def relationship_exists(driver, database, id1, id2):
    query = """
    MATCH (a:TextChunk {id: $id1})-[r:SIMILAR_TO]->(b:TextChunk {id: $id2})
    RETURN r
    """
    parameters = {"id1": id1, "id2": id2}
    with driver.session(database=database) as session:
        result = session.run(query, parameters)
        return result.single() is not None
  

def create_relationships(driver, database, df, dataset_name):
    existing_chunks = get_all_text_chunks(driver, database)
    new_embeddings = df['embedding'].tolist()
    new_ids = [f"{dataset_name}_{i}" for i in range(len(new_embeddings))]

    all_chunks = existing_chunks + list(zip(new_ids, new_embeddings))
    all_embeddings = [embedding for _, embedding in all_chunks]
    similarity_matrix = cosine_similarity(all_embeddings)
    
    for i in range(len(existing_chunks), len(all_chunks)):
        for j in range(len(all_chunks)):
            if i == j:
                continue
            similarity_score = similarity_matrix[i][j]
            if similarity_score > 0.8:
                id1 = all_chunks[i][0]
                id2 = all_chunks[j][0]
                if id1 != id2 and not relationship_exists(driver, database, id1, id2):
                    query = """
                    MATCH (a:TextChunk {id: $id1}), (b:TextChunk {id: $id2})
                    CREATE (a)-[:SIMILAR_TO {score: $score}]->(b)
                    """
                    parameters = {
                        "id1": id1,
                        "id2": id2,
                        "score": similarity_score
                    }
                    execute_query(driver, database, query, parameters)


def process_dataset(driver, database, dataset_path, dataset_name):
    print(f"\nProcessing dataset: {dataset_name} into: {database}", "\n")
    df = pd.read_parquet(dataset_path)
    create_nodes(driver, database, df, dataset_name)
    create_relationships(driver, database, df, dataset_name)


def main():
    parser = argparse.ArgumentParser(description='Populate Neo4j graph with data from a specific Parquet dataset.')
    parser.add_argument('--dataset', type=str, required=True, help='Base name for the input Parquet dataset, no need to add .parquet extension.')
    parser.add_argument('--database', type=str, required=True, help='Name of the Neo4j database to connect to.')

    args = parser.parse_args()
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    database = args.database
    driver = GraphDatabase.driver(uri, auth=(username, password))

    dataset_name = args.dataset if args.dataset.endswith('.parquet') else f'{args.dataset}.parquet'
    dataset_path = f'./datasets/{dataset_name}'

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"The dataset: {dataset_path} does not exist. Please check your path and try again.")

    process_dataset(driver, database, dataset_path, dataset_name)

    driver.close()

if __name__ == "__main__":
    main()