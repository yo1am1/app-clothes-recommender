from typing import Any

import uvicorn
import logging
import pathlib
import pandas as pd
from fastapi import FastAPI, HTTPException

import chromadb
from chromadb.utils.embedding_functions.open_clip_embedding_function import (
    OpenCLIPEmbeddingFunction,
)
from chromadb.utils.data_loaders import ImageLoader

from utils.vectorstore_worker import recommend_for_user

DATA_DIR = pathlib.Path(__file__).parent / "data"
USER_PROFILES_PATH = DATA_DIR / "user_profiles.csv"
PURCHASE_HISTORY_PATH = DATA_DIR / "purchase_history.csv"
IMAGES_FOLDER = DATA_DIR / "images"

embedding_function = OpenCLIPEmbeddingFunction()
data_loader = ImageLoader()

chroma_client = chromadb.PersistentClient(path=str(DATA_DIR / "chroma"))
multimodal_collection = chroma_client.get_or_create_collection(
    name="multimodal_images",
    embedding_function=embedding_function,
    data_loader=data_loader,
)


def load_user_profiles() -> pd.DataFrame:
    """
    Load user profiles from the CSV file.
    :return: DataFrame containing user profiles.
    """
    if not USER_PROFILES_PATH.exists():
        logging.warning(f"User profiles file not found: {USER_PROFILES_PATH}")
        return pd.DataFrame()
    return pd.read_csv(USER_PROFILES_PATH)


def load_purchase_history() -> pd.DataFrame:
    """
    Load purchase history from the CSV file.
    :return:  DataFrame containing purchase history.
    """
    if not PURCHASE_HISTORY_PATH.exists():
        logging.warning(f"Purchase history file not found: {PURCHASE_HISTORY_PATH}")
        return pd.DataFrame()
    return pd.read_csv(PURCHASE_HISTORY_PATH)


app = FastAPI(
    title="Clothes Recommender API",
    version="1.0.0",
    description="Basic API for recommending clothes to users.",
)


@app.on_event("startup")
def on_startup():
    """
    Load necessary data at startup and store them in app.state.
    """
    logging.info("Loading user profiles...")
    app.state.user_data = load_user_profiles()

    logging.info("Loading purchase history...")
    app.state.purchase_data = load_purchase_history()

    logging.info("Startup tasks completed.")


@app.get("/recommend/{user_id}")
def recommend_endpoint(user_id: str, top_n: int = 5):
    """
    Returns top_n recommended product IDs and their image paths for the given user_id.

    :param user_id: User ID for which to generate recommendations.
    :param top_n: Number of recommendations to return.
    :return: JSON response containing user_id, recommended_ids, metadata_filter, and image_paths.
    """
    user_data = app.state.user_data
    purchase_data = app.state.purchase_data

    if user_data.empty:
        raise HTTPException(status_code=404, detail="No user data loaded.")
    if purchase_data.empty:
        raise HTTPException(status_code=404, detail="No purchase history loaded.")

    # Use your recommend_for_user function
    rec_ids, rec_metas, rec_filter = recommend_for_user(
        collection=multimodal_collection,
        user_id=user_id,
        user_data=user_data,
        purchase_data=purchase_data,
        top_n=top_n,
    )

    # Build file paths (assuming each image is stored as "<product_id>.jpg" in IMAGES_FOLDER)
    image_paths = [str(IMAGES_FOLDER / f"{pid}.jpg") for pid in rec_ids]

    return {
        "user_id": user_id,
        "recommended_ids": rec_ids,
        "metadata_filter": rec_filter,
        "image_paths": image_paths,
    }


if __name__ == "__main__":
    uvicorn.run("fastapi_app:app", host="127.0.0.1", port=8000, reload=True)
