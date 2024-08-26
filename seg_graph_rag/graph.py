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


def check_node_exists(driver, database, text, dataset_name):
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
        if not check_node_exists(driver, database, row['chunk'], dataset_name):
            query = """
            MATCH (d:Dataset {name: $dataset})
            CREATE (t:TextChunk {id: $id, text: $text, text_embedding: $text_embedding, dataset: $dataset})
            CREATE (d)-[:CONTAINS]->(t)
            """
            parameters = {
                "id": unique_id,
                "text": row['chunk'],
                "text_embedding": row['embedding'],
                "dataset": dataset_name
            }
            execute_query(driver, database, query, parameters)


def get_all_text_chunks(driver, database):
    query = """
    MATCH (t:TextChunk)
    RETURN t.id AS id, t.text_embedding AS text_embedding
    """
    with driver.session(database=database) as session:
        result = session.run(query)
        chunks = [(record["id"], record["text_embedding"]) for record in result]
    return chunks


def check_relationship_exists(driver, database, id1, id2, relationship_type):
    query = f"""
    MATCH (a:TextChunk {{id: $id1}})-[r:{relationship_type}]->(b:TextChunk {{id: $id2}})
    RETURN r
    """
    parameters = {"id1": id1, "id2": id2}
    with driver.session(database=database) as session:
        result = session.run(query, parameters)
        return result.single() is not None


def create_similarity_relationships_text_embeddings(driver, database, df, dataset_name, existing_chunks=None):
    if existing_chunks is None:
        existing_chunks = get_all_text_chunks(driver, database)
        
    new_embeddings = df['embedding'].tolist()
    new_ids = [f"{dataset_name}_{i}" for i in range(len(new_embeddings))]

    for new_id, new_embedding in zip(new_ids, new_embeddings):
        all_chunks = existing_chunks + [(new_id, new_embedding)]
        all_embeddings = [embedding for _, embedding in all_chunks]
        similarity_matrix = cosine_similarity([new_embedding], all_embeddings)[0]

        for j, similarity_score in enumerate(similarity_matrix):
            if similarity_score > 0.8:
                id2 = all_chunks[j][0]
                if new_id != id2 and not check_relationship_exists(driver, database, new_id, id2, "SIMILAR_TO_TEXT"):
                    query = """
                    MATCH (a:TextChunk {id: $id1}), (b:TextChunk {id: $id2})
                    CREATE (a)-[:SIMILAR_TO_TEXT {score: $score}]->(b)
                    """
                    parameters = {
                        "id1": new_id,
                        "id2": id2,
                        "score": similarity_score
                    }
                    execute_query(driver, database, query, parameters)

        existing_chunks.append((new_id, new_embedding))


def create_node_embeddings(driver, database):
    query = """
    CALL gds.graph.project('textGraph', 'TextChunk', 'SIMILAR_TO_TEXT', {
        nodeProperties: ['text_embedding'],
        relationshipProperties: []
    })
    """
    execute_query(driver, database, query)

    query = """
    CALL gds.node2vec.stream('textGraph', {
        embeddingDimension: 1536,
        iterations: 10,
        walkLength: 80
    })
    YIELD nodeId, embedding
    RETURN gds.util.asNode(nodeId).id AS id, embedding
    """
    with driver.session(database=database) as session:
        result = session.run(query)
        embeddings = {record["id"]: record["embedding"] for record in result}

    for node_id, embedding in embeddings.items():
        query = """
        MATCH (t:TextChunk {id: $id})
        SET t.node_embedding = $node_embedding
        """
        parameters = {"id": node_id, "node_embedding": embedding}
        execute_query(driver, database, query, parameters)

    query = """
    CALL gds.graph.drop('textGraph', false)
    """
    execute_query(driver, database, query)


def create_similarity_relationships_node_embeddings(driver, database):
    query = """
    MATCH (t:TextChunk)
    RETURN t.id AS id, t.node_embedding AS node_embedding
    """
    with driver.session(database=database) as session:
        result = session.run(query)
        all_chunks = [(record["id"], record["node_embedding"]) for record in result]

    all_embeddings = [embedding for _, embedding in all_chunks]
    similarity_matrix = cosine_similarity(all_embeddings)
    
    for i in range(len(all_chunks)):
        for j in range(len(all_chunks)):
            if i == j:
                continue
            similarity_score = similarity_matrix[i][j]
            if similarity_score > 0.8:
                id1 = all_chunks[i][0]
                id2 = all_chunks[j][0]
                if id1 != id2 and not check_relationship_exists(driver, database, id1, id2, "SIMILAR_TO_NODES"):
                    query = """
                    MATCH (a:TextChunk {id: $id1}), (b:TextChunk {id: $id2})
                    CREATE (a)-[:SIMILAR_TO_NODES {score: $score}]->(b)
                    """
                    parameters = {
                        "id1": id1,
                        "id2": id2,
                        "score": similarity_score
                    }
                    execute_query(driver, database, query, parameters)


def process_dataset(driver, database, dataset_path, dataset_name, batch_size=None):
    print(f"\nProcessing dataset: {dataset_name} into: {database}\n")

    df = pd.read_parquet(dataset_path)
    
    if batch_size:
        print(f"Processing in batches of {batch_size} rows", "\n")
        num_batches = len(df) // batch_size + (1 if len(df) % batch_size > 0 else 0)
        existing_chunks = get_all_text_chunks(driver, database)
        
        for batch_num in range(num_batches):
            batch_df = df.iloc[batch_num * batch_size:(batch_num + 1) * batch_size]
            print(f"Processing batch {batch_num + 1} of {num_batches}")
            create_nodes(driver, database, batch_df, dataset_name)
            create_similarity_relationships_text_embeddings(driver, database, batch_df, dataset_name, existing_chunks)

            existing_chunks.extend([(f"{dataset_name}_{i}", row['embedding']) for i, row in batch_df.iterrows()])
    else:
        print("Processing entire dataset at once", "\n")
        create_nodes(driver, database, df, dataset_name)
        create_similarity_relationships_text_embeddings(driver, database, df, dataset_name)

    create_node_embeddings(driver, database)
    create_similarity_relationships_node_embeddings(driver, database)


def main():
    parser = argparse.ArgumentParser(description='Populate Neo4j graph with data from a specific Parquet dataset.')
    parser.add_argument('--dataset', type=str, required=True, help='Base name for the input Parquet dataset, no need to add .parquet extension.')
    parser.add_argument('--database', type=str, required=True, help='Name of the Neo4j database to connect to.')
    parser.add_argument('--batch', type=int, help='Number of rows to process per batch.')

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

    process_dataset(driver, database, dataset_path, dataset_name, args.batch)

    driver.close()

if __name__ == "__main__":
    main()