import argparse
import os
import fitz
import pyarrow as pa
import pyarrow.parquet as pq
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter


def generate_embeddings(text):
    client = OpenAI()
    model = "text-embedding-3-small"
    response = client.embeddings.create(
        input=text,
        model=model
    )
    embeddings = response.data[0].embedding
    return embeddings


def process_pdf(file_path, output_name):
    text_splitter = RecursiveCharacterTextSplitter()
    doc = fitz.open(file_path)
    chunk_texts = []
    embeddings = []

    for page in doc:
        blocks = page.get_text("blocks")
        page_blocks = {}

        for x0, y0, x1, y1, block_text, block_no, block_type in blocks:
            if block_type == 0:  # Only process text blocks
                if block_no not in page_blocks:
                    page_blocks[block_no] = [block_text]
                else:
                    page_blocks[block_no].append(block_text)

        for block_no in sorted(page_blocks.keys()):
            block_text = "\n".join(page_blocks[block_no]).replace('\n', ' ').strip()
            if block_text:
                processed_chunks = text_splitter.split_text(block_text)
                for chunk_text in processed_chunks:
                    embedding = generate_embeddings(chunk_text)
                    
                    print(f"Chunk:\n{chunk_text}\n")
                    print(f"Embedding:\n{embedding}\n")
                    
                    chunk_texts.append(chunk_text)
                    embeddings.append(embedding)

    new_table = pa.table({
        'chunk': chunk_texts,
        'embedding': embeddings
    })

    if not os.path.exists('./datasets'):
        os.makedirs('./datasets')

    output_path = f'./datasets/{output_name}'
    if os.path.exists(output_path):
        existing_table = pq.read_table(output_path)
        combined_table = pa.concat_tables([existing_table, new_table])
    else:
        combined_table = new_table

    pq.write_table(combined_table, output_path)


def main():
    parser = argparse.ArgumentParser(description='Process PDF files into chunks, generate embeddings using OpenAI, and store chunks/embeddings in a Parquet dataset.')
    parser.add_argument('--document', type=str, required=True, help='Full path to the PDF file to process.')
    parser.add_argument('--dataset', type=str, required=True, help='Base name for the output Parquet dataset, no need to add .parquet extension.')

    args = parser.parse_args()
    
    if not os.path.exists(args.document):
        raise FileNotFoundError(f"The path/PDF document: {args.document} does not exist. Please check your path and try again.")

    output_name = f'{args.dataset}.parquet' if not args.dataset.endswith('.parquet') else args.dataset
    process_pdf(args.document, output_name)

if __name__ == "__main__":
    main()