import streamlit as st
import pathlib
import os
import hashlib
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter

import chromadb
from chromadb.utils.embedding_functions.open_clip_embedding_function import (
    OpenCLIPEmbeddingFunction,
)
from chromadb.utils.data_loaders import ImageLoader
from icecream import ic


def format_metadata(meta: dict) -> str:
    """Formats metadata into a caption string."""
    if not meta:
        return "No metadata"

    product_name = meta.get("product_name", "Unknown")
    price = meta.get("price", "N/A")
    brand = meta.get("brand_name", "Unknown")
    category = meta.get("product_category", "Unknown")

    return f"{product_name}\n{brand} | {category}\nPrice: {price}"


def remove_duplicate_ids_with_metadata(ids, metadatas):
    """Remove duplicates while preserving association of IDs and metadata."""
    seen = set()
    unique_ids = []
    unique_metas = []

    for _id, meta in zip(ids, metadatas):
        if _id not in seen:
            seen.add(_id)
            unique_ids.append(_id)
            unique_metas.append(meta)

    return unique_ids, unique_metas


def load_images_from_ids(
    image_ids, images_folder, captions_list=None, remove_identical_files=True
):
    """
    Loads images from disk given a list of product IDs.
    Assumes each image is stored as "<id>.jpeg" in images_folder.
    If captions_list is provided, uses those as captions.
    """
    images = []
    loaded_captions = []
    loaded_hashes = set()

    for idx, image_id in enumerate(image_ids):
        image_path = os.path.join(images_folder, f"{image_id}.jpeg")

        if os.path.exists(image_path):
            try:
                img = Image.open(image_path)

                if remove_identical_files:
                    img_hash = hashlib.md5(img.tobytes()).hexdigest()
                    if img_hash in loaded_hashes:
                        continue
                    loaded_hashes.add(img_hash)

                images.append(img)

                if captions_list and idx < len(captions_list):
                    loaded_captions.append(captions_list[idx])
                else:
                    loaded_captions.append(image_id)
            except Exception as e:
                st.error(f"Error loading {image_id}: {e}")

    return images, loaded_captions


def load_product_metadata_dict(csv_path: str) -> dict:
    """
    Reads preprocessed.csv and builds a dictionary mapping product_id (extracted from image_path)
    to a formatted metadata string.
    """
    metadata_dict = {}

    if not os.path.exists(csv_path):
        return metadata_dict

    df = pd.read_csv(csv_path)

    for row in df.itertuples():
        image_path = getattr(row, "image_path", None)
        if image_path and isinstance(image_path, str):
            base = os.path.basename(image_path)
            product_id = os.path.splitext(base)[0]
            meta = {
                "product_name": getattr(row, "product_name", None),
                "price": getattr(row, "price", None),
                "brand_name": getattr(row, "brand_name", None),
                "product_category": getattr(row, "product_category", None),
                "description": getattr(row, "description", None),
                "style_attributes": getattr(
                    row, "stely_attributes", None
                ),  # note: adjust spelling if needed
                "available_size": getattr(row, "available_size", None),
                "color": getattr(row, "color", None),
            }
            metadata_dict[product_id] = format_metadata(meta)

    return metadata_dict


def search_similar_images(collection, query: str, top_k: int = 20):
    result = collection.query(
        query_texts=[query], n_results=top_k, include=["distances", "metadatas"]
    )

    return result


def search_by_image(collection, image_array: np.ndarray, top_k: int = 20):
    result = collection.query(
        query_images=[image_array], n_results=top_k, include=["distances", "metadatas"]
    )

    ic(result)

    ids_list = result.get("ids", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    combined = list(zip(ids_list, metadatas, distances))
    combined_sorted = sorted(combined, key=lambda x: x[2])  # sort by distance ascending

    # Unpack sorted values
    sorted_ids = [item[0] for item in combined_sorted]
    sorted_metas = [item[1] for item in combined_sorted]

    return sorted_ids, sorted_metas


def build_metadata_filter(known_colors=None, known_sizes=None):
    """
    Returns a metadata filter dict for Chroma that will match if the product's metadata
    has a color or available_size within the provided lists.

    The filter uses the "$or" operator, for example:

    {
      "$or": [
         {"color": {"$in": known_colors}},
         {"available_size": {"$in": known_sizes}}
      ]
    }

    If neither list is provided, returns None.
    """
    conditions = []

    if known_colors:
        conditions.append({"color": {"$in": known_colors}})
    if known_sizes:
        conditions.append({"available_size": {"$in": known_sizes}})

    if conditions:
        return {"$or": conditions}
    return None


def recommend_for_user(
    collection,
    user_id: str,
    user_data: pd.DataFrame,
    purchase_data: pd.DataFrame,
    top_n=5,
):
    """
    A recommendation function that:
      1) Retrieves user preferences.
      2) Builds a text query from user preferences.
      3) Constructs a metadata filter based on the user's purchase history:
         - Colors: Top 3 most frequent (normalized) colors from the purchase history.
         - Sizes: All unique (normalized) sizes from the purchase history.
         If both colors and sizes exist, the filter is constructed with a top-level "$or" operator.
         If neither exists, no filter is applied.
      4) Queries the vector store with the text query and the metadata filter (if available).
      5) Excludes items the user already purchased.
      6) Returns top_n recommended product IDs, their metadata, and the metadata filter used.
    """
    # Retrieve user profile row
    row = user_data[user_data["user_id"] == int(user_id)]
    if row.empty:
        st.warning(
            f"No user profile found for user_id={user_id}. Returning random items."
        )
        result = collection.query(
            query_texts=["lingerie"], n_results=20, include=["metadatas", "distances"]
        )
        candidate_ids = result.get("ids", [[]])[0]
        return candidate_ids[:top_n], [], None

    # Parse user preferences from profile
    preferences_str = row["preferences"].values[0]
    if isinstance(preferences_str, str) and preferences_str.startswith("["):
        import ast

        try:
            preferences_list = ast.literal_eval(preferences_str)
        except Exception as e:
            preferences_list = []
    elif isinstance(preferences_str, str):
        preferences_list = [preferences_str]
    else:
        preferences_list = []
    user_query = " ".join(preferences_list) if preferences_list else "lingerie"

    # Retrieve purchase history for the user
    user_purchases = purchase_data[purchase_data["user_id"] == int(user_id)]

    # If no purchase history exists, omit the metadata filter (return random items)
    if user_purchases.empty:
        metadata_filter = None
    else:
        from collections import Counter

        # Normalize color and size values
        colors = (
            user_purchases["color"]
            .dropna()
            .astype(str)
            .apply(lambda x: x.strip().title())
            .tolist()
        )
        sizes = (
            user_purchases["size"]
            .dropna()
            .astype(str)
            .apply(lambda x: x.strip())
            .tolist()
        )
        unique_sizes = list(set(sizes))
        # Get the top 3 most frequent colors
        if colors:
            color_counts = Counter(colors)
            top_colors = [color for color, count in color_counts.most_common(3)]
        else:
            top_colors = []

        # Build metadata filter using a top-level $or operator if both exist.
        if unique_sizes and top_colors:
            metadata_filter = {
                "$or": [
                    {"available_size": {"$in": unique_sizes}},
                    {"color": {"$in": top_colors}},
                ]
            }
        elif unique_sizes:
            metadata_filter = {"available_size": {"$in": unique_sizes}}
        elif top_colors:
            metadata_filter = {"color": {"$in": top_colors}}
        else:
            metadata_filter = None

    # Query the vector store, applying the metadata filter if available
    if metadata_filter:
        results = collection.query(
            query_texts=[user_query],
            n_results=20,
            include=["metadatas", "distances"],
            where=metadata_filter,
        )
    else:
        results = collection.query(
            query_texts=[user_query], n_results=20, include=["metadatas", "distances"]
        )

    candidate_ids = results.get("ids", [[]])[0]
    candidate_metas = results.get("metadatas", [[]])[0]

    # Exclude products the user already purchased
    purchased_ids = set(user_purchases["product_id"].astype(str).tolist())
    recommended_ids = []
    recommended_metas = []
    for _id, meta in zip(candidate_ids, candidate_metas):
        if _id not in purchased_ids:
            recommended_ids.append(_id)
            recommended_metas.append(meta)
        if len(recommended_ids) == top_n:
            break

    if not recommended_ids:
        st.warning(
            "No new items found. Possibly user already owns everything in top search results."
        )

    return recommended_ids, recommended_metas, metadata_filter
