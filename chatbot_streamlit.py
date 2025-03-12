__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3

import streamlit as st
import pathlib
import numpy as np
import pandas as pd
from PIL import Image
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)

import chromadb
from chromadb.utils.embedding_functions.open_clip_embedding_function import (
    OpenCLIPEmbeddingFunction,
)
from chromadb.utils.data_loaders import ImageLoader

from utils.embeddings.azure_embeddings import __get_openai_model
from utils.prompts import init_message
from utils.vectorstore_worker import (
    load_images_from_ids,
    load_product_metadata_dict,
    recommend_for_user,
    format_metadata,
    remove_duplicate_ids_with_metadata,
    search_by_image,
    search_similar_images,
)
from icecream import ic

DATA_DIR = pathlib.Path(__file__).parent / "data"
USER_PROFILES_PATH = DATA_DIR / "user_profiles.csv"
PURCHASE_HISTORY_PATH = DATA_DIR / "purchase_history.csv"
PREPROCESSED_CSV = DATA_DIR / "preprocessed_normalized.csv"
IMAGES_FOLDER = DATA_DIR / "images"

embedding_function = OpenCLIPEmbeddingFunction()
data_loader = ImageLoader()

chroma_client = chromadb.PersistentClient(
    path=str(pathlib.Path(__file__).parent / "data" / "chroma")
)
multimodal_db = chroma_client.get_or_create_collection(
    name="multimodal_images",
    embedding_function=embedding_function,
    data_loader=data_loader,
)

metadata_field_info = [
    AttributeInfo(
        name="brand_name",
        description="The brand name of the product. Example: Wacoal, Calvin Klein, Aerie, Hanky Panky, Victorias Secret, Topshop, B Temptd, Victorias Secret Pink, Btemptd By Wacoal, Vanity Fair, Nordstrom Lingerie, Calvin Klein Modern Cotton, Calvin Klein Performance, Btemptd, Aeo, Creative Motion or Unknown.",
        type="string",
    ),
    AttributeInfo(
        name="color",
        description="The color of the product. Example: Black, White, Orange, etc.",
        type="string",
    ),
    AttributeInfo(
        name="price",
        description="The price of the product",
        type="number",
    ),
    AttributeInfo(
        name="available_size",
        description="The available sizes for the product",
        type="string",
    ),
]


@st.cache_data(show_spinner=False)
def load_user_profiles() -> pd.DataFrame:
    """
    Load user profiles from user_profiles.csv.
    :return: DataFrame of user profiles.
    """
    if not USER_PROFILES_PATH.exists():
        st.warning(f"User profiles file not found: {USER_PROFILES_PATH}")
        return pd.DataFrame()
    return pd.read_csv(USER_PROFILES_PATH)


@st.cache_data(show_spinner=False)
def load_purchase_history() -> pd.DataFrame:
    """
    Load purchase history from purchase_history.csv.
    :return: DataFrame of purchase history.
    """
    if not PURCHASE_HISTORY_PATH.exists():
        st.warning(f"Purchase history file not found: {PURCHASE_HISTORY_PATH}")
        return pd.DataFrame()
    return pd.read_csv(PURCHASE_HISTORY_PATH)


def __add_message() -> None:
    """
    Add user message to the chat history.
    :return:  None
    """
    st.session_state.messages.append(
        {"role": "user", "content": st.session_state.get("chat_input", None)}
    )


def __session_init() -> None:
    """
    Initialize the session state.
    :return: None
    """
    if st.session_state.get("messages", None) is None:
        st.session_state.messages = [
            {"role": "ai", "content": "Hi! I am Clothy. How can I help you?"}
        ]


def __get_response(chain, user_input, context_str, history_str):
    """
    Get the response from the model chain.
    :param chain:  The model chain.
    :param user_input:  The user input.
    :param context_str:  The context.
    :param history_str:  The history.
    :return:  The response from the model chain.
    """
    return chain.stream(
        {
            "query": user_input,
            "context": context_str,
            "history": history_str,
        }
    )


def __build_metadata_filter(filter_obj) -> dict:
    """
    Recursively build a metadata filter dictionary for ChromaDB from a filter object.
    If no valid filter is built, returns an empty dictionary.
    :param filter_obj:  The filter object.
    :return:  The metadata filter dictionary.
    """
    if filter_obj is None:
        return {}

    if hasattr(filter_obj, "operator"):
        op_name = filter_obj.operator.name.lower()  # e.g., "and" or "or"
        sub_filters = [__build_metadata_filter(arg) for arg in filter_obj.arguments]
        sub_filters = [f for f in sub_filters if f]

        if not sub_filters:
            return {}

        if op_name == "and":
            return sub_filters[0] if len(sub_filters) == 1 else {"$and": sub_filters}
        elif op_name == "or":
            return sub_filters[0] if len(sub_filters) == 1 else {"$or": sub_filters}
        else:
            return {}

    else:
        comparator = filter_obj.comparator
        comp_name = (
            comparator.name.lower()
            if hasattr(comparator, "name")
            else str(comparator).lower()
        )

        attribute = filter_obj.attribute
        value = filter_obj.value

        if comp_name == "eq":
            return {attribute: {"$eq": value}}
        elif comp_name == "gt":
            return {attribute: {"$gt": value}}
        elif comp_name == "lt":
            return {attribute: {"$lt": value}}
        elif comp_name == "gte":
            return {attribute: {"$gte": value}}
        elif comp_name == "lte":
            return {attribute: {"$lte": value}}
        else:
            return {}


def init_page() -> None:
    """
    Initialize the page layout.
    :return: None
    """
    st.title("Clothes Recommender System")

    user_data = load_user_profiles()
    purchase_data = load_purchase_history()
    __session_init()

    # Tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Text to Image Search", "Search by Image", "User Recommendations", "Chatbot"]
    )

    with tab1:
        st.markdown(
            """
            ## Text to Image Search

            This section allows you to find products by entering a text-based query. 
            For example, if you type "black bra," the system will search for the 
            most visually and semantically similar products in the database. 
            It then displays up to five matching items, along with their IDs and 
            any available metadata (like product name, price, and brand).

            **How it works:**
            1. You enter a text query in the box below.
            2. When you click "Search by Text," the system uses a text embedding model 
               to convert your query into a vector.
            3. It searches a vector database (ChromaDB) to find product images and metadata 
               that best match your query.
            4. The top matches are displayed along with images (if available).
            """
        )

        k_results = st.number_input(
            "Enter the amount of results to display",
            value=5,
            min_value=1,
            max_value=20,
            step=1,
            key="k_results",
        )

        query = st.text_input("Enter your search query", value="black bra")
        if st.button("Search by Text"):
            with st.spinner("Searching..."):
                result = search_similar_images(
                    query=query,
                    top_k=st.session_state.k_results,
                    collection=multimodal_db,
                )

                raw_ids = result.get("ids", [[]])[0]
                raw_metas = result.get("metadatas", [[]])[0]

                unique_ids, unique_metas = remove_duplicate_ids_with_metadata(
                    raw_ids, raw_metas
                )

                top_ids = unique_ids[: st.session_state.k_results]
                top_metas = unique_metas[: st.session_state.k_results]

                captions = [format_metadata(meta) for meta in top_metas]
                st.write("Found image IDs:", top_ids)
                images, loaded_captions = load_images_from_ids(
                    top_ids, str(IMAGES_FOLDER), captions_list=captions
                )
                if images:
                    st.image(images, caption=loaded_captions, width=200)
                else:
                    st.warning("No images found or failed to load images.")

    with tab2:
        st.markdown(
            """
            ## Search by Image

            This section allows you to find products by uploading an image. 
            For example, if you have a photo of a bra, the system will analyze 
            the image and retrieve visually similar items from our database. 

            **How it works:**
            1. You upload an image (PNG/JPG/JPEG) via the file uploader below.
            2. The system converts the uploaded image into an embedding (a vector representation).
            3. It searches a vector database (ChromaDB) for items with similar embeddings.
            4. The top matches are displayed below, along with their IDs and any available metadata.
            """
        )

        uploaded_image = st.file_uploader(
            "Upload an image", type=["png", "jpg", "jpeg"]
        )
        if uploaded_image and st.button("Search by Image"):
            with st.spinner("Searching..."):
                img = np.array(Image.open(uploaded_image).convert("RGB"))
                image_ids, image_metas = search_by_image(
                    image_array=img, top_k=5, collection=multimodal_db
                )
                unique_ids, unique_metas = remove_duplicate_ids_with_metadata(
                    image_ids, image_metas
                )

                top_ids = unique_ids[:5]
                top_metas = unique_metas[:5]

                captions = [format_metadata(meta) for meta in top_metas]
                st.write("Found image IDs:", top_ids)
                images, loaded_captions = load_images_from_ids(
                    top_ids, str(IMAGES_FOLDER), captions_list=captions
                )

                if images:
                    st.image(images, caption=loaded_captions, width=200)
                else:
                    st.warning("No images found or failed to load images.")

    with tab3:
        st.markdown(
            """
            ## User Recommendations

            This tab provides personalized product recommendations for each user, 
            based on their purchase history and stated preferences. 
            Below, you'll see the full purchase history, including product IDs, 
            ratings, timestamps, sizes, and colors. 
            You can select a User ID from the dropdown to generate recommendations 
            tailored to that specific user. The system will:

            1. Load user profiles and purchase history.
            2. Identify the user's preferences (e.g., favorite brands, sizes, or colors).
            3. Use a recommendation algorithm (including any filters, embeddings, or 
               other logic) to find products the user is most likely to be interested in.
            4. Display the recommended product IDs, along with any relevant metadata 
               and product images (if available).

            Simply choose a User ID and click "Get Recommendations" to see the 
            top recommended products for that user.
            """
        )
        st.table(data=purchase_data)

        user_ids = (
            user_data["user_id"].drop_duplicates().astype(str).tolist()
            if not user_data.empty
            else []
        )
        selected_user_id = (
            st.selectbox("Select a User ID", options=user_ids) if user_ids else "None"
        )

        if st.button("Get Recommendations"):
            if selected_user_id == "None":
                st.warning("No user IDs found in user_profiles.csv!")
            else:
                with st.spinner("Generating recommendations..."):
                    captions = []

                    rec_ids, rec_metas, rec_filter = recommend_for_user(
                        user_id=selected_user_id,
                        user_data=user_data,
                        purchase_data=purchase_data,
                        top_n=5,
                        collection=multimodal_db,
                    )
                    st.write("Recommended product IDs:", rec_ids)
                    st.write("Recommended metadata:", rec_filter)

                    product_meta_dict = load_product_metadata_dict(
                        str(PREPROCESSED_CSV)
                    )

                    for pid in rec_ids:
                        cap = product_meta_dict.get(pid, pid)
                        captions.append(cap)

                    images, loaded_captions = load_images_from_ids(
                        rec_ids, str(IMAGES_FOLDER), captions_list=captions
                    )

                    if images:
                        st.image(images, caption=loaded_captions, width=200)
                    else:
                        st.warning(
                            "No images found for the recommended IDs (or images failed to load)."
                        )

    with tab4:
        st.markdown(
            """
            ## Chatbot and Self-Query Retrieval

            This chatbot can assist you with questions about our clothing products.
            You can also use the self-query retrieval feature to search for products
            using a natural language query that leverages metadata (e.g., brand, color,
            size, price). Simply enter your query in the field below.
            """
        )
        chat_model = __get_openai_model()

        document_content_description = "Brief summary of the product on Amazon."
        prompt = get_query_constructor_prompt(
            document_content_description,
            metadata_field_info,
        )
        output_parser = StructuredQueryOutputParser.from_components()
        query_constructor = prompt | chat_model | output_parser
        msg_chain = init_message.model_copy() | chat_model | StrOutputParser()

        with st.container():
            if st.session_state["messages"]:
                for messages in st.session_state["messages"]:
                    if messages["role"] == "ai":
                        with st.chat_message(name="assistant"):
                            st.write(messages["content"])
                    elif messages["role"] == "user":
                        with st.chat_message(name="human"):
                            st.write(messages["content"])

        user_input = st.chat_input(
            "Talk to Clothy...", key="chat_input", on_submit=__add_message
        )

        if user_input:
            structured_query = query_constructor.invoke({"query": user_input})
            ic(structured_query.filter)

            metadata_filter = __build_metadata_filter(structured_query.filter)
            if not metadata_filter:
                metadata_filter = None  # Pass None if filter is empty.

            ic(metadata_filter)
            ic(structured_query.query)

            filtered_context = multimodal_db.query(
                query_texts=[user_input],
                n_results=5,
                include=["distances", "metadatas"],
                where=metadata_filter,
            )
            ic(filtered_context)

            with st.chat_message(name="assistant"), st.spinner("Thinking..."):
                model_response = st.write_stream(
                    __get_response(
                        chain=msg_chain,
                        user_input=user_input,
                        context_str=filtered_context
                        if filtered_context is not None
                        else "No context found.",
                        history_str=st.session_state["messages"],
                    )
                )
            if model_response:
                st.session_state.messages.append(
                    {"role": "ai", "content": model_response}
                )
            st.json(st.session_state)


if __name__ == "__main__":
    init_page()
