import os
import re
import whisper
import json
import asyncio
from typing import List, Dict
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent
from datetime import datetime
from pymongo import MongoClient
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Load Environment ---
load_dotenv()

# --- Load Whisper Model ---
whisper_model = whisper.load_model("base")

# --- MongoDB Client ---
client = MongoClient(os.getenv("MONGO_URI"))
mongo_db = os.getenv("MONGO_DB")
if mongo_db is None:
    raise ValueError("MONGO_DB environment variable is not set")
db = client[mongo_db]

# --- Recipe Data Class ---
class Recipe(BaseModel):
    ingredients: Dict[str, int]
    instructions: str

def load_recipes_from_db() -> Dict[str, Recipe]:
    collection = db["items"]  # Use the correct collection name: "items"

    recipes = {}
    for doc in collection.find({"isAvailable": True}):
        try:
            name = doc["name"].lower()

            # Parse ingredients (list of dicts with name, quantity, unit)
            ingredients = {}
            for ing in doc.get("ingredients", []):
                ing_name = ing["name"].lower()
                quantity = ing["quantity"]
                ingredients[ing_name] = quantity  # Note: ignoring unit for now

            instructions = doc.get("instructions") or doc.get("description") or "No instructions provided."

            recipes[name] = Recipe(ingredients=ingredients, instructions=instructions)
        except Exception as e:
            print(f"Error loading item '{doc.get('name')}': {e}")
    return recipes

# --- Recipe Database ---
recipe_db: Dict[str, Recipe] = load_recipes_from_db()

def refresh_recipe_cache():
    global recipe_db
    recipe_db = load_recipes_from_db()

# --- Helper Functions ---
def classify_intent(text: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You're a restaurant assistant. "
                        "Classify the user's query into one of the following intents:\n"
                        "- get_recipe\n"
                        "- check_inventory\n"
                        "- get_trending\n"
                        "- most_requested\n"
                        "- profit_suggestion\n"
                        "If the intent is unclear, return: unknown"
                    ),
                },
                {"role": "user", "content": text}
            ]
        )

        content = response.choices[0].message.content
        if content:
            result = content.strip().lower()
            if result in {"get_recipe", "check_inventory", "get_trending", "most_requested", "profit_suggestion"}:
                return result
        return "unknown"

    except Exception as e:
        print("Error classifying intent:", e)
        return "unknown"

def extract_dish_name(text: str) -> str:
    for dish in recipe_db:
        if dish.lower() in text.lower():
            return dish
    return ""

def check_inventory_availability(dish: str, inventory: Dict[str, int]) -> str:
    if dish not in recipe_db:
        return f"Sorry, I don't have the recipe for {dish}."

    missing = []
    for ingredient in recipe_db[dish].ingredients:
        closest = find_closest_ingredient(ingredient, list(inventory.keys()))
        required = recipe_db[dish].ingredients[ingredient]
        available = inventory.get(closest, 0)
        if available < required:
            missing.append(ingredient)
    if missing:
        return f"To prepare {dish}, you're missing: {', '.join(missing)}."
    return f"Yes, you have all ingredients for {dish}."

def get_trending_recipes() -> str:
    return "Lemon juice, Omelette, and Grilled Cheese are trending today."

def log_serving(dish_name: str, recipe: Recipe, inventory: Dict[str, int]):
    usage = []
    for ing, amount in recipe.ingredients.items():
        if ing in inventory:
            usage.append({"name": ing, "quantity": amount})
    db["serving_logs"].insert_one({
        "dish_name": dish_name,
        "ingredients_used": usage,
        "timestamp": datetime.utcnow()
    })

def log_query(user_text: str, dish_name: str):
    db["query_logs"].insert_one({
        "user_query": user_text,
        "dish_mentioned": dish_name,
        "timestamp": datetime.utcnow()
    })

import requests
from bs4 import BeautifulSoup

def get_recipe_from_web(dish_name: str) -> Recipe | None:
    search_url = f"https://www.allrecipes.com/search?q={dish_name.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        page = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(page.content, "html.parser")

        # Find first recipe link
        recipe_link = soup.select_one("a.card__titleLink") or soup.select_one("a[data-internal-referrer-link='hub recipe']")
        if not recipe_link:
            return None

        if recipe_link and recipe_link.has_attr("href"):
            recipe_url = str(recipe_link["href"])
        else:
            return None  # Or fallback

        recipe_page = requests.get(recipe_url, headers=headers)
        recipe_soup = BeautifulSoup(recipe_page.content, "html.parser")

        # Extract ingredients and instructions
        ingredients = {}
        for li in recipe_soup.select("span.ingredients-item-name"):
            name = li.get_text(strip=True).lower()
            ingredients[name] = 1  # You can’t get exact quantity easily, so default to 1

        steps = [step.get_text(strip=True) for step in recipe_soup.select("ul.instructions-section li p")]
        return Recipe(
            ingredients=ingredients,
            instructions="\n".join(steps)
        )

    except Exception as e:
        print(f"Failed to get recipe from web: {e}")
        return None

from difflib import get_close_matches

from difflib import get_close_matches

def find_closest_ingredient(name: str, inventory_keys: List[str]) -> str:
    matches = get_close_matches(name.lower(), inventory_keys, n=1, cutoff=0.7)
    return matches[0] if matches else name

from datetime import datetime, timedelta

def get_recent_trending(days=3):
    since = datetime.utcnow() - timedelta(days=days)
    pipeline = [
        {"$match": {"timestamp": {"$gte": since}}},
        {"$group": {"_id": "$dish_mentioned", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}, {"$limit": 3}
    ]
    return [f"{r['_id']} ({r['count']} requests)" for r in db["query_logs"].aggregate(pipeline)]


# --- Main Agent Logic ---
async def restaurant_agent(user_text: str, inventory: Dict[str, int], is_audio: bool = False):
    intent = classify_intent(user_text)
    dish = extract_dish_name(user_text)

    if dish:
        log_query(user_text, dish)

    if intent == "get_recipe":
        if dish in recipe_db:
            log_serving(dish, recipe_db[dish], inventory)
            return recipe_db[dish].instructions
        else:
            recipe = get_recipe_from_web(dish)
            if recipe:
                # Optionally log or add to db
                return f"I found a recipe online for {dish}:\n\n{recipe.instructions}"
            return f"Sorry, I couldn’t find a recipe for {dish}."

    elif intent == "check_inventory":
        return check_inventory_availability(dish, inventory)

    elif intent == "get_trending":
        return get_trending_recipes()

    elif intent == "most_requested":
        pipeline = [
            {"$group": {"_id": "$dish_mentioned", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 1}
        ]
        top = list(db["query_logs"].aggregate(pipeline))
        if top:
            return f"The most requested dish is '{top[0]['_id']}' with {top[0]['count']} requests."
        else:
            return "No dish request data available yet."

    elif intent == "profit_suggestion":
        pipeline = [
            {"$group": {"_id": "$dish_name", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 3}
        ]
        top_served = list(db["serving_logs"].aggregate(pipeline))
        suggestions = []
        for entry in top_served:
            dish = entry["_id"]
            recipe = recipe_db.get(dish)
            if not recipe:
                continue
            missing = [i for i in recipe.ingredients if i not in inventory]
            suggestions.append(f"Dish: {dish} (served {entry['count']}x) - Needs: {', '.join(missing) or 'All in stock'}")
        return "To maximize profit, focus on:\n" + "\n".join(suggestions)
    
    elif intent == "get_trending":
        return get_recent_trending()

    return "I can help with food-related topics, ingredients, and recipes. Is there anything specific you'd like to know?"

# --- Audio Handler ---
def transcribe_audio(file_path: str) -> str:
    if not file_path.endswith((".mp3", ".wav", ".m4a")):
        raise ValueError("Unsupported audio file format.")
    if os.path.getsize(file_path) > 10 * 1024 * 1024:  # 10MB limit
        raise ValueError("Audio file too large.")

    result = whisper_model.transcribe(file_path)
    text = result.get("text")
    if not isinstance(text, str):
        raise TypeError(f"Unexpected return type for result['text']: {type(text)}")
    return text

# --- Entry Point ---
async def assistant_query(input_data: str, inventory: Dict[str, int], recipe_db_input: Dict[str, Recipe], is_audio=False):
    global recipe_db
    recipe_db = recipe_db_input  # Update recipes if needed

    if is_audio:
        try:
            transcription = transcribe_audio(input_data)
            return await restaurant_agent(transcription, inventory, is_audio=True)
        except Exception as e:
            return f"Audio processing failed: {str(e)}"

    else:
        return await restaurant_agent(input_data, inventory, is_audio=False)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Text or audio input")
    parser.add_argument("--audio", action="store_true", help="Indicates audio input")
    args = parser.parse_args()

    # Example inventory
    example_inventory = {"lemon": 3, "water": 2, "sugar": 1, "egg": 4, "salt": 2, "oil": 2}

    result = asyncio.run(assistant_query(args.text, example_inventory, recipe_db, is_audio=args.audio))
    print("\nAssistant Response:\n", result)
