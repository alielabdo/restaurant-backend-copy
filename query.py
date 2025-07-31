import sys, asyncio, argparse
from assistant import assistant_query, recipe_db
from pymongo import MongoClient
from dotenv import load_dotenv
import os

def fetch_inventory_from_db():
    load_dotenv()
    uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB")
    collection_name = os.getenv("MONGO_COLLECTION")

    client = MongoClient(uri)
    collection = client[db_name][collection_name]
    inventory = {}
    for item in collection.find():
        name = item.get("name", "").lower()
        qty = item.get("quantity", 0)
        inventory[name] = qty
    return inventory

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", help="Text question from user")
    parser.add_argument("--audio", help="Path to audio file (voice input)")
    args = parser.parse_args()

    inventory = fetch_inventory_from_db()

    if args.audio:
        result = asyncio.run(assistant_query(args.audio, inventory, recipe_db, is_audio=True))
    elif args.text:
        result = asyncio.run(assistant_query(args.text, inventory, recipe_db, is_audio=False))
    else:
        print("No input provided", file=sys.stderr)
        sys.exit(1)

    print(result)

if __name__ == "__main__":
    main()
