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

# --- Load Environment ---
load_dotenv()

# --- Load Whisper Model ---
whisper_model = whisper.load_model("base")

# --- MongoDB Client ---
client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("MONGO_DB")]

# --- Recipe Data Class ---
class Recipe(BaseModel):
    ingredients: Dict[str, int]
    instructions: str

# --- Recipe Database ---
recipe_db: Dict[str, Recipe] = {
    "lemon juice": Recipe(
        ingredients={"lemon": 2, "water": 1, "sugar": 1},
        instructions="Squeeze the lemons. Mix juice with water and sugar. Stir well."
    ),
    "omelette": Recipe(
        ingredients={"egg": 2, "salt": 1, "oil": 1},
        instructions="Beat eggs, add salt. Heat oil in pan. Pour eggs. Cook both sides."
    )
}

# --- Helper Functions ---
def classify_intent(text: str) -> str:
    text = text.lower()
    if any(q in text for q in ["how to", "prepare", "make", "cook"]):
        return "get_recipe"
    elif any(q in text for q in ["available", "do i have", "is there"]):
        return "check_inventory"
    elif any(q in text for q in ["trending", "popular now", "what's hot"]):
        return "get_trending"
    elif any(q in text for q in ["most requested", "most popular", "top dish"]):
        return "most_requested"
    elif any(q in text for q in ["maximize profit", "what should i sell", "make more money"]):
        return "profit_suggestion"
    return "unknown"

def extract_dish_name(text: str) -> str:
    for dish in recipe_db:
        if dish.lower() in text.lower():
            return dish
    return ""

def check_inventory_availability(dish: str, inventory: Dict[str, int]) -> str:
    if dish not in recipe_db:
        return f"Sorry, I don't have the recipe for {dish}."
    missing = [i for i in recipe_db[dish].ingredients if i not in inventory or inventory[i] < recipe_db[dish].ingredients[i]]
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
            return f"Sorry, I don't have the recipe for {dish}."

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

    return "I can help with food-related topics, ingredients, and recipes. Is there anything specific you'd like to know?"

# --- Audio Handler ---
def transcribe_audio(file_path: str) -> str:
    result = whisper_model.transcribe(file_path)
    return result["text"]

# --- Entry Point ---
async def assistant_query(input_data: str, inventory: Dict[str, int], recipe_db_input: Dict[str, Recipe], is_audio=False):
    global recipe_db
    recipe_db = recipe_db_input  # Update recipes if needed

    if is_audio:
        transcription = transcribe_audio(input_data)
        return await restaurant_agent(transcription, inventory, is_audio=True)
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
