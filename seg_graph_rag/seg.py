import argparse
import os
import fitz
import nltk
from nltk.corpus import stopwords
import re
import pyarrow as pa
import pyarrow.parquet as pq
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import base64


nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\b\w{1,2}\b', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text


def generate_embeddings(client, text):
    model = "text-embedding-3-small"
    response = client.embeddings.create(
        input=text,
        model=model
    )
    embeddings = response.data[0].embedding
    return embeddings


def generate_image_summary(client, image_data):
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image concisely."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                        },
                    },
                ],
            }
        ],
        max_tokens=100,
    )
    return "Image: " + response.choices[0].message.content


def process_pdf(client, file_path, output_name, process_images=False):
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
                    cleaned_text = clean_text(chunk_text)
                    embedding = generate_embeddings(client, cleaned_text)
                    print(f"Chunk:\n{chunk_text}\n")
                    print(f"Embedding:\n{embedding}\n")
                    if embedding is not None:
                        chunk_texts.append(cleaned_text)
                        embeddings.append(embedding)

        # Process images
        if process_images:
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_data = base_image["image"]
                
                image_summary = generate_image_summary(client, image_data)
                print(f"Image Summary:\n{image_summary}\n")
                
                image_embedding = generate_embeddings(client, image_summary)
                
                if image_embedding is not None:
                    chunk_texts.append(image_summary)
                    embeddings.append(image_embedding)

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
    parser.add_argument('--images', action='store_true', help='Send images to OpenAI for summary.')

    args = parser.parse_args()
    client = OpenAI()
    
    if not os.path.exists(args.document):
        raise FileNotFoundError(f"The path/PDF document: {args.document} does not exist. Please check your path and try again.")

    output_name = f'{args.dataset}.parquet' if not args.dataset.endswith('.parquet') else args.dataset
    process_pdf(client, args.document, output_name, process_images=args.images)

if __name__ == "__main__":
    main()