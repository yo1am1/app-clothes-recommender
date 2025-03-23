# utils/image_embedding.py
import pathlib
import os
import numpy as np
from PIL import Image
import chromadb
from chromadb.utils.embedding_functions.open_clip_embedding_function import (
    OpenCLIPEmbeddingFunction,
)
from chromadb.utils.data_loaders import ImageLoader
from icecream import ic
from tqdm import tqdm
import pandas as pd

# Initialize Chroma client and collection.
DATA_DIR = pathlib.Path(__file__).parent.parent.parent / "data"
chroma_client = chromadb.PersistentClient(path=str(DATA_DIR / "chroma"))

embedding_function = OpenCLIPEmbeddingFunction()
data_loader = ImageLoader()

multimodal_db = chroma_client.get_or_create_collection(
    name="multimodal_images",
    embedding_function=embedding_function,
    data_loader=data_loader,
)


def add_images_with_metadata(csv_path, images_folder):
    """
    Reads a CSV file containing product information and the corresponding image filename,
    then adds each image to the collection along with its metadata.
    """
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Optionally, filter only rows with a valid image path
    df = df[df["image_path"].notna()].copy()

    for row in tqdm(df.itertuples(), total=len(df), desc="Adding images with metadata"):
        # Construct the full image path using the CSV column (assumed to be "image_path")
        image_file = getattr(row, "image_path", None)
        if not image_file:
            continue
        full_image_path = os.path.join(images_folder, os.path.basename(image_file))
        if not os.path.exists(full_image_path):
            print(f"Image not found: {full_image_path}")
            continue

        try:
            # Load image and convert to RGB
            image = np.array(Image.open(full_image_path).convert("RGB"))
        except Exception as e:
            print(f"Error opening image {full_image_path}: {e}")
            continue

        # Build metadata dictionary from CSV fields (adjust field names if needed)
        metadata = {
            "product_name": getattr(row, "product_name", None),
            "price": getattr(row, "price", None),
            "brand_name": getattr(row, "brand_name", None),
            "product_category": getattr(row, "product_category", None),
            "description": getattr(row, "description", None),
            "style_attributes": getattr(row, "style_attributes", None),
            "available_size": getattr(row, "available_size", None),
            "color": getattr(row, "color", None),
        }

        # Use a unique ID for each document (here we use the CSV index)
        doc_id = str(row.Index)

        try:
            multimodal_db.add(ids=[doc_id], images=[image], metadatas=[metadata])
        except Exception as e:
            print(f"Error indexing image {full_image_path} with metadata: {e}")


def search_similar_images(query):
    """
    Searches the Chroma collection by text and prints the result (including distances and metadata).
    """
    result = multimodal_db.query(
        query_texts=[query], include=["distances", "metadatas"]
    )
    ic(result)


if __name__ == "__main__":
    csv_path = str(DATA_DIR / "preprocessed_normalized.csv")
    images_folder = str(DATA_DIR / "images")
    add_images_with_metadata(csv_path, images_folder)

    search_similar_images("Give me black underwear")
