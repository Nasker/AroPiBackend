import csv
import os
import requests
from opensymbols_api import query_symbol
from db_utils import create_schema, insert_picto
from config import DB_PATH, IMAGES_DIR

def download_image(url, word):
    ext = url.split('.')[-1].split('?')[0]
    filename = f"{word}.{ext}"
    path = os.path.join(IMAGES_DIR, filename)

    if not os.path.exists(path):
        img = requests.get(url, timeout=10)
        img.raise_for_status()
        with open(path, "wb") as f:
            f.write(img.content)

    return path

def main():
    print("Creating DB schema...")
    create_schema(DB_PATH)

    with open("../input/words.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            word = row["word"].strip()
            lang = row["language"].strip()
            pref = row["preferred_source"].strip()
            grammar_type = row.get("grammar_type", "").strip()

            print(f"Searching: {word}")

            symbol = query_symbol(word, pref)

            if not symbol:
                print(f"No results for {word}")
                continue

            img_url = symbol["image_url"]
            image_path = download_image(img_url, word)

            entry = {
                "word": word,
                "language": lang,
                "grammar_type": grammar_type,
                "source": symbol.get("repo_key"),
                "symbol_id": symbol.get("id"),
                "keywords": symbol.get("search_string", ""),
                "image_path": image_path
            }

            insert_picto(DB_PATH, entry)
            print(f"Added {word}")

    print("Done!")

if __name__ == "__main__":
    main()
