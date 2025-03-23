"""This script loads and preprocesses raw data, including removing duplicates,
null values, and downloading product images from Amazon. It supports a '--force'
argument to replace existing data, a '--cont' flag to continue from the last downloaded image,
and a '--validate' flag to validate and update the database with image paths and download state.
"""

import hashlib
import os
import argparse
import logging
import pathlib
import requests

import pandas as pd
from PIL import Image
from bs4 import BeautifulSoup
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Constants
DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
IMAGE_DIR = DATA_DIR / "images"
PROCESSED_DATA_PATH = DATA_DIR / "preprocessed_normalized.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    :return: argparse.Namespace object with parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process clothing data.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing by replacing existing data",
    )
    parser.add_argument(
        "--cont",
        action="store_true",
        help="Continue processing from the last downloaded image; do not remove old data.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate that images listed in the dataset exist and update CSV with image path and download state.",
    )
    return parser.parse_args()


def compute_image_hash(image_path: str) -> str:
    """Compute the MD5 hash of an image file.

    :param image_path: Path to the image file.
    :return: MD5 hash of the image file.
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            image_bytes = img.tobytes()
            return hashlib.md5(image_bytes).hexdigest()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def remove_duplicate_images(directory: str) -> None:
    """
    Iterates through all images in a directory (and subdirectories),
    and removes duplicates by comparing MD5 hashes.

    :param directory: Path to the directory containing images.
    :return: None
    """
    seen_hashes = {}  # Maps hash -> file path
    duplicates = []  # List of duplicate file paths

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".jpeg")):
                file_path = os.path.join(root, file)
                img_hash = compute_image_hash(file_path)
                if img_hash is None:
                    continue
                if img_hash in seen_hashes:
                    print(
                        f"Duplicate found:\n  {file_path}\n  is a duplicate of\n  {seen_hashes[img_hash]}"
                    )
                    duplicates.append(file_path)
                else:
                    seen_hashes[img_hash] = file_path

    for dup_path in duplicates:
        try:
            os.remove(dup_path)
            print(f"Removed duplicate file: {dup_path}")
        except Exception as e:
            print(f"Error removing {dup_path}: {e}")


def remove_path_from_dataset(csv_file: str, images_directory: str) -> None:
    """
    Updates the dataset CSV file by checking each row's 'image_path'.
    If the file does not exist in the images_directory,
    sets 'image_path' to None and 'image_downloaded' to False.

    :param csv_file: Path to the CSV file containing the dataset.
    :param images_directory: Path to the directory containing the images.
    :return: None
    """
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file {csv_file}: {e}")
        return

    updated = False

    for idx, row in df.iterrows():
        image_path = row.get("image_path")
        if pd.notna(image_path):
            abs_path = image_path
            if not os.path.isabs(image_path):
                abs_path = os.path.join(images_directory, image_path)
            if not os.path.exists(abs_path):
                print(
                    f"Image not found for row {idx}: {image_path}. Updating dataset entry."
                )
                df.at[idx, "image_path"] = None
                df.at[idx, "image_downloaded"] = False
                updated = True

    if updated:
        df.to_csv(csv_file, index=False)
        print("Dataset updated with removed image paths.")
    else:
        print("No changes made to the dataset.")


def load_and_clean_data() -> pd.DataFrame:
    """
    Loads raw CSV files from data/raw, merges them into a single DataFrame,
    drops duplicate URLs, removes null rows, and returns the cleaned DataFrame.
    """
    dfs = []

    logging.info("Loading raw data files...")
    for file in RAW_DATA_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            logging.error(f"Failed to read {file}: {e}")

    if not dfs:
        logging.warning("No CSV files found in raw data directory.")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    logging.info(f"Loaded {len(dfs)} files. Dataset shape: {df.shape}")

    if "pdp_url" in df.columns:
        df.drop_duplicates(subset=["pdp_url"], keep="first", inplace=True)

    df.dropna(how="any", inplace=True)
    df.reset_index(drop=True, inplace=True)

    logging.info(f"Data cleaned. New shape: {df.shape}")

    return df


def download_image(url: str, save_path: pathlib.Path) -> bool:
    """Downloads an image from a given URL and saves it as JPEG format."""
    try:
        response = requests.get(url, headers=HEADERS, stream=True, timeout=10)

        if response.status_code == 200:
            temp_path = save_path.with_suffix(".tmp")

            with open(temp_path, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)

            with Image.open(temp_path) as img:
                rgb_img = img.convert("RGB")
                rgb_img.save(save_path, "JPEG", quality=95)

            temp_path.unlink()

            return True
    except requests.RequestException as e:
        logging.error(f"Error downloading image from {url}: {e}")
    except Exception as e:
        logging.error(f"Error processing image as JPEG from {url}: {e}")

    return False


def extract_image_url(product_url: str) -> str | None:
    """
    Extracts the main product image URL from an Amazon product page,
    by scraping the 'landingImage'.
    """
    try:
        response = requests.get(product_url, headers=HEADERS, timeout=10)

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.content, "html.parser")
        img_tag = soup.find("img", {"id": "landingImage"})

        if not img_tag:
            return None

        return img_tag.get("data-old-hires") or img_tag.get("src")
    except requests.RequestException as e:
        logging.error(f"Error fetching product page {product_url}: {e}")

        return None


def download_images(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downloads product images from their 'pdp_url' (if it's an Amazon link)
    and resumes from the last downloaded image.

    :param df: DataFrame with 'pdp_url' column containing product URLs.
    :return: Updated DataFrame with 'image_path' and 'image_downloaded' columns.
    """
    logging.info("Starting image download process...")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    max_existing_idx = -1

    for img_file in IMAGE_DIR.glob("*.jpeg"):
        try:
            idx_int = int(img_file.stem)
            max_existing_idx = max(max_existing_idx, idx_int)
        except ValueError:
            pass
    if "image_path" not in df.columns:
        df["image_path"] = None
    if "image_downloaded" not in df.columns:
        df["image_downloaded"] = False

    seen_urls = set()
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading images"):
        if idx <= max_existing_idx:
            continue

        product_url = row["pdp_url"]

        if product_url in seen_urls:
            df.at[idx, "image_downloaded"] = False
            continue

        seen_urls.add(product_url)

        if "amazon" in product_url.lower():
            img_url = extract_image_url(product_url)

            if img_url:
                image_path = IMAGE_DIR / f"{idx}.jpeg"

                if not image_path.exists():
                    success = download_image(img_url, image_path)
                    df.at[idx, "image_path"] = str(image_path) if success else None
                    df.at[idx, "image_downloaded"] = success
                else:
                    logging.info(f"Image already exists locally: {image_path}")
                    df.at[idx, "image_path"] = str(image_path)
                    df.at[idx, "image_downloaded"] = True
            else:
                df.at[idx, "image_downloaded"] = False
        else:
            df.at[idx, "image_downloaded"] = False
    logging.info("Image downloading completed.")
    return df


def save_data(df: pd.DataFrame) -> None:
    """Saves the cleaned and processed DataFrame to a CSV file."""
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    logging.info(f"Processed data saved to {PROCESSED_DATA_PATH}")


def check_existing_data():
    """
    Checks whether preprocessed data or images exist so we don't repeat the process unnecessarily.
    """
    data_exists = PROCESSED_DATA_PATH.exists()
    images_exist = any(IMAGE_DIR.glob("*.jpeg"))

    if data_exists:
        logging.info(f"Existing processed data found: {PROCESSED_DATA_PATH}")
    else:
        logging.warning("Processed data file not found.")

    if images_exist:
        logging.info("Some images already exist in the images folder.")
    else:
        logging.warning("No images found in the images folder.")

    return data_exists, images_exist


def delete_existing_data():
    """Deletes existing preprocessed data and images if '--force' is used."""
    if PROCESSED_DATA_PATH.exists():
        PROCESSED_DATA_PATH.unlink()
        logging.info("Deleted existing preprocessed data file.")

    for image in IMAGE_DIR.glob("*.jpeg"):
        image.unlink()

    logging.info("Deleted existing images.")


def validate_and_update_images(csv_file: str, images_directory: str):
    """
    Validates the dataset based on the images available in the directory.
    Iterates over all image files in the folder, uses the filename (without extension)
    as the record ID, and updates the CSV:
      - If an image exists and can be opened, its absolute path is stored and 'image_downloaded' is set to True.
      - If no image is found or it is invalid, the record is updated accordingly.
    Finally, the CSV is saved with the updated values and a summary is printed.

    :param csv_file: Path to the CSV file containing the dataset.
    :param images_directory: Path to the directory containing the images.
    :return: None
    """
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file {csv_file}: {e}")
        return

    # Build a mapping from record ID (string) to absolute image path from the images directory.
    image_map = {}
    for img_file in pathlib.Path(images_directory).glob("*.jpeg"):
        try:
            record_id = img_file.stem  # e.g., "5406" from "5406.jpeg"
            image_map[record_id] = str(img_file.resolve())
        except Exception as e:
            print(f"Error processing file {img_file}: {e}")
            continue

    valid_count = 0
    invalid_count = 0

    total_rows = len(df)
    print(f"Total records in CSV: {total_rows}")

    # Assume that the row index (as a string) corresponds to the record ID in the images folder.
    for idx, row in df.iterrows():
        rec_id = str(idx)

        if rec_id in image_map:
            abs_path = image_map[rec_id]

            try:
                with Image.open(abs_path) as img:
                    img.verify()  # Verify that image can be opened

                df.at[idx, "image_path"] = abs_path
                df.at[idx, "image_downloaded"] = True

                valid_count += 1
            except Exception as e:
                print(f"Record {idx}: Image at {abs_path} is invalid. Error: {e}")

                df.at[idx, "image_path"] = None
                df.at[idx, "image_downloaded"] = False

                invalid_count += 1
        else:
            # No image found for this record
            df.at[idx, "image_path"] = None
            df.at[idx, "image_downloaded"] = False

            invalid_count += 1

    try:
        df.to_csv(csv_file, index=False)

        print(f"Total images in directory: {len(image_map)}")
        print(
            f"CSV updated: {valid_count} valid images, {invalid_count} invalid/missing images."
        )
    except Exception as e:
        print(f"Error writing updated CSV file {csv_file}: {e}")


def main():
    """
    Main function to execute the full data preprocessing pipeline:
      1) Parse arguments for --force, --cont, and --validate.
      2) If '--validate' is used, validate existing images and update CSV, then exit.
      3) Check existing data.
      4) If '--force' is used, remove old data.
      5) If neither '--force' nor '--cont' is used and data exists, do nothing.
      6) Otherwise, load & clean data, remove duplicates.
      7) Download images from unique URLs (resuming from the last downloaded image).
      8) Save final CSV with image paths.
      9) Remove duplicate images.
      10) Update CSV for any missing image files.
    """
    args = parse_args()

    if args.validate:
        print("Validating and updating image paths in the dataset...")
        validate_and_update_images(str(PROCESSED_DATA_PATH), str(IMAGE_DIR))
        return

    data_exists, images_exist = check_existing_data()

    if args.force:
        logging.info("Forcing re-processing: deleting existing data...")
        delete_existing_data()
    elif not args.cont:
        if data_exists or images_exist:
            logging.info(
                "Data already exists. Use '--force' to re-process or '--cont' to continue processing."
            )
            return

    df = load_and_clean_data()
    if df.empty:
        logging.warning("No data available to process.")
        return

    df = download_images(df)
    save_data(df)

    input(
        "This script will remove duplicate images in the directory. Press Enter to continue..."
    )

    remove_duplicate_images(str(IMAGE_DIR))
    remove_path_from_dataset(str(PROCESSED_DATA_PATH), str(IMAGE_DIR))

    logging.info(
        "Data processing completed. Removed duplicates and updated dataset with valid image paths."
    )


if __name__ == "__main__":
    main()
