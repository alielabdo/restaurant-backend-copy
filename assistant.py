import os
import re
import json
import requests
from typing import List, Dict, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from pydantic_ai import Agent
import whisper

# --- Init Agent ---
load_dotenv()
model = "google-gla:gemini-2.0-flash"
system_prompt = """
You are a restaurant AI assistant. Only respond to questions related to food, ingredients, recipes, and restaurant operations.
When providing recipes, be detailed with ingredients and steps. For web results, summarize key information.
Reject anything off-topic.
"""
agent = Agent(model=model, system_prompt=system_prompt)

# Web search setup (using SerpAPI - you can use any search API)
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

async def search_web_for_recipe(query: str) -> Optional[Dict]:
    """Search the web for a recipe using SerpAPI"""
    if not SERPAPI_KEY:
        return None
        
    params = {
        "q": f"{query} recipe",
        "hl": "en",
        "gl": "us",
        "api_key": SERPAPI_KEY
    }
    
    try:
        response = requests.get("https://serpapi.com/search", params=params)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Web search error: {e}")
    return None

def parse_web_recipe_result(search_result: Dict) -> Dict:
    """Extract recipe information from web search results"""
    recipes = []
    
    # Parse knowledge graph if available
    if "knowledge_graph" in search_result:
        kg = search_result["knowledge_graph"]
        if "recipe" in kg.get("description", "").lower():
            recipes.append({
                "title": kg.get("title", ""),
                "ingredients": kg.get("ingredients", []),
                "steps": kg.get("steps", []),
                "source": "Knowledge Graph"
            })
    
    # Parse organic results
    for result in search_result.get("organic_results", []):
        if "recipe" in result.get("title", "").lower():
            recipes.append({
                "title": result.get("title", ""),
                "link": result.get("link", ""),
                "snippet": result.get("snippet", ""),
                "source": "Web Result"
            })
    
    return {"recipes": recipes} if recipes else None

import asyncio
from google.genai.errors import ServerError

async def safe_run_agent(prompt, output_type=str):
    for attempt in range(3):  # retry max 3 times
        try:
            result = await agent.run(prompt, output_type=output_type)
            return result.output
        except ServerError as e:
            print(f"Gemini server busy, retrying... ({attempt+1}/3)")
            await asyncio.sleep(2)  # Wait a bit before retry
    return "Sorry, the AI service is currently unavailable. Please try again shortly."

# --- Audio Transcription ---
whisper_model = whisper.load_model("base")
def transcribe_audio(path: str) -> str:
    return whisper_model.transcribe(path)["text"]

# --- Data Models ---
class Recipe(BaseModel):
    name: str
    ingredients: Dict[str, str]
    steps: List[str]
    dietary_info: Dict[str, bool] = {
        "vegetarian": False,
        "vegan": False,
        "gluten_free": False
    }

class RecipeResponse(BaseModel):
    recipe_name: str
    full_ingredients: List[str]
    missing_ingredients: List[str]
    preparation_steps: List[str]

# --- Core Functions ---
def is_restaurant_related(text: str) -> bool:
    text = text.lower()
    
    # Expanded keyword sets
    food_keywords = {
        "food", "ingredient", "recipe", "cook", "kitchen", "dish", "menu", 
        "prepare", "make", "bake", "grill", "boil", "fry", "juice", "drink",
        "meal", "serve", "snack", "appetizer", "dessert", "beverage", "soda"
    }
    
    preparation_phrases = {
        "how to make", "how do i make", "how to prepare", "how do i prepare",
        "steps to make", "recipe for", "way to cook", "method for",
        "what's in a", "what goes into"
    }
    
    # Check for preparation phrases first
    if any(phrase in text for phrase in preparation_phrases):
        return True
        
    # Check for ingredient mentions
    ingredient_pattern = r"\b(lemon|sugar|water|salt|pepper|flour|rice|cheese|meat|vegetable|fruit)\b"
    if re.search(ingredient_pattern, text):
        return True
        
    # Check general food keywords
    tokens = re.findall(r'\w+', text)
    return any(token in food_keywords for token in tokens)

def classify_intent(text: str):
    text = text.lower()
    if any(phrase in text for phrase in ["how do i make", "how to prepare", "how do i prepare"]):
        return "get_recipe"
    elif "what can i make" in text:
        return "match_inventory"
    elif "popular" in text or "suggest" in text:
        return "trending"
    return "unknown"

def extract_dish_name(text: str) -> str:
    text = re.sub(r"[^\w\s]", "", text.lower())
    for phrase in ["how do i prepare", "how to prepare", "how do i make", "how to make", "prepare", "make"]:
        if phrase in text:
            return text.split(phrase)[-1].strip()
    return text.strip()

def fetch_recipe(dish_name: str, db: Dict[str, Recipe]) -> Recipe:
    return db.get(dish_name.lower())

from typing import NamedTuple

class RecipeMatch(NamedTuple):
    full_ingredients: list[str]
    missing_ingredients: list[str]
    preparation_steps: list[str]

# Map of ingredient aliases (synonyms)
INGREDIENT_ALIASES = {
    "lemon": ["lemon juice", "lemon fruit", "fresh lemon", "lemon wedge"],
    "sugar": ["white sugar", "granulated sugar", "cane sugar", "table sugar"],
    "water": ["filtered water", "bottled water", "mineral water", "sparkling water"],
    "ice": ["ice cubes", "crushed ice", "ice block"],
    "bread": ["baguette", "french bread", "italian bread", "dinner rolls"],
    "garlic": ["garlic cloves", "fresh garlic", "minced garlic", "garlic powder"],
    "butter": ["unsalted butter", "salted butter", "margarine"],
    # Add more comprehensive aliases
}

def match_inventory(recipe, inventory_db: dict) -> RecipeMatch:
    inventory_keys = {k.lower(): v for k, v in inventory_db.items()}
    full, missing = [], []

    for ingredient in recipe.ingredients.keys():  # Now using .keys() since ingredients is a dict
        normalized = ingredient.lower()
        has_ingredient = False

        # Direct match
        if normalized in inventory_keys and inventory_keys[normalized] > 0:
            has_ingredient = True
        else:
            # Check aliases
            aliases = INGREDIENT_ALIASES.get(normalized, [])
            aliases += [k for k, v in INGREDIENT_ALIASES.items() if normalized in v]

            for alias in aliases:
                if alias in inventory_keys and inventory_keys[alias] > 0:
                    has_ingredient = True
                    break

        if has_ingredient:
            full.append(f"{ingredient} ({recipe.ingredients[ingredient]})")
        else:
            missing.append(f"{ingredient} ({recipe.ingredients[ingredient]})")

    return RecipeMatch(
        full_ingredients=full,
        missing_ingredients=missing,
        preparation_steps=recipe.steps
    )

def get_trending_recipe() -> Recipe:
    return Recipe(
        name="Lemon Ice Juice",
        ingredients={
            "lemon": "2-3",
            "ice": "1 cup",
            "sugar": "1/2 cup",
            "water": "4 cups"
        },
        steps=["Squeeze lemons", "Mix with water & sugar", "Add ice"],
        dietary_info={
            "vegetarian": True,
            "vegan": True,
            "gluten_free": True
        }
    )

def classify_intent(text: str):
    text = text.lower()
    
    if any(phrase in text for phrase in ["how do i make", "how to prepare", "recipe for"]):
        return "get_recipe"
    elif any(phrase in text for phrase in ["what can i make", "what to make with"]):
        return "match_inventory"
    elif any(phrase in text for phrase in ["popular", "suggest", "recommend", "trending"]):
        return "trending"
    elif any(phrase in text for phrase in ["help", "what can you do"]):
        return "help"
    return "unknown"

async def generate_recipe(dish_name: str, inventory_db: dict) -> str:
    """Generate a recipe using the LLM, considering available ingredients"""
    prompt = f"""
    The user asked how to prepare '{dish_name}'. 
    They have these ingredients available: {list(inventory_db.keys())}
    
    Please:
    1. Create a practical recipe including:
       - List of ingredients with quantities (prioritizing what they have)
       - Step-by-step preparation instructions
       - Estimated preparation time
    2. Mention any missing ingredients they would need to buy
    3. Provide serving suggestions
    
    Make the recipe clear and easy to follow for home cooking.
    """
    return await safe_run_agent(prompt, output_type=str)

async def restaurant_agent(user_text: str, inventory_db: dict, recipe_db: dict):
    if not is_restaurant_related(user_text):
        return await safe_run_agent(
            f"The user said: '{user_text}'. Respond politely that you only answer food and restaurant-related questions.",
            output_type=str
        )

    intent = classify_intent(user_text)

    if intent == "get_recipe":
        dish = extract_dish_name(user_text)
        recipe = fetch_recipe(dish, recipe_db)
        
        if recipe:
            # Handle local recipe with inventory check
            response_data = match_inventory(recipe, inventory_db)
            
            prompt = f"""
            The user asked how to prepare '{recipe.name}'.
            Here's their inventory: {list(inventory_db.keys())}
            
            Recipe Details:
            - Ingredients: {', '.join(f"{k} ({v})" for k, v in recipe.ingredients.items())}
            - Missing Ingredients: {', '.join(response_data.missing_ingredients) if response_data.missing_ingredients else 'None'}
            - Steps: {'; '.join(recipe.steps)}
            
            Provide:
            1. Confirmation of the recipe name
            2. Clear list of needed ingredients
            3. Note about missing items (if any)
            4. Numbered preparation steps
            5. Serving suggestions
            """
            return await safe_run_agent(prompt, output_type=str)
        else:
            # Generate recipe dynamically
            try:
                return await generate_recipe(dish, inventory_db)
            except Exception as e:
                print(f"Error generating recipe: {e}")
                return await safe_run_agent(
                    f"I couldn't create a recipe for '{dish}'. Please try a different dish or ask something else.",
                    output_type=str
                )

    elif intent == "match_inventory":
        matched, missing_info = [], []
        for recipe in recipe_db.values():
            missing = [i for i in recipe.ingredients if i.lower() not in map(str.lower, inventory_db.keys())]
            if not missing:
                matched.append(recipe.name)
            else:
                missing_info.append((recipe.name, missing))

        prompt = f"""
        The user has this inventory: {list(inventory_db.keys())}.
        Recipes they can make: {matched}
        Recipes they can't make: {', '.join(f"{name} (missing: {', '.join(missing)})" for name, missing in missing_info)}

        Be clear and helpful in your response. Mention missing ingredients where needed.
        """
        return await safe_run_agent(prompt, output_type=str)

    elif intent == "trending":
        recipe = get_trending_recipe()
        response_data = match_inventory(recipe, inventory_db)
        prompt = f"""
        The user asked for a trending recipe. Here's a suggestion:
        - Name: {recipe.name}
        - Ingredients: {recipe.ingredients}
        - Missing ingredients: {response_data.missing_ingredients}
        - Preparation steps: {recipe.steps}

        First inform the user whether they can make this with their current inventory.
        If anything is missing, list it clearly.
        Then explain how to prepare the dish.
        """
        return await safe_run_agent(prompt, output_type=str)

    else:
        return await safe_run_agent(
            "The user asked something I could not classify. Politely ask them to rephrase.",
            output_type=str
        )

# --- Query Interface ---
def process_audio_query(audio_path: str, inventory_db: dict, recipe_db: dict):
    transcribed_text = transcribe_audio(audio_path)
    print(f"Transcribed: {transcribed_text}")
    return restaurant_agent(transcribed_text, inventory_db, recipe_db)

async def assistant_query(input_data, inventory_db: dict, recipe_db: dict, is_audio=False):
    user_text = transcribe_audio(input_data) if is_audio else input_data
    return await restaurant_agent(user_text, inventory_db, recipe_db)

# --- Dummy Inventory Import ---
import json

with open("inventory.json", "r") as f:
    inventory = json.load(f)

# --- Sample Recipe DB ---
# --- Sample Recipe DB ---
recipe_db = {
    "lemon ice juice": Recipe(
        name="Lemon Ice Juice",
        ingredients={
            "lemon": "2-3",
            "ice": "1 cup",
            "sugar": "1/2 cup",
            "water": "4 cups"
        },
        steps=[
            "Juice 2-3 lemons to get about 1/2 cup of lemon juice",
            "In a pitcher, mix the lemon juice with 4 cups of cold water",
            "Add 1/2 cup of sugar (adjust to taste) and stir until dissolved",
            "Add 1 cup of ice cubes and stir",
            "Serve chilled with lemon slices for garnish"
        ],
        dietary_info={
            "vegetarian": True,
            "vegan": True,
            "gluten_free": True
        }
    ),
    "classic lemonade": Recipe(
        name="Classic Lemonade",
        ingredients={
            "lemon": "6",
            "water": "5 cups (1 for syrup, 4 for mixing)",
            "sugar": "1 cup"
        },
        steps=[
            "Juice 6 lemons to get about 1 cup of lemon juice",
            "Make simple syrup by heating 1 cup water with 1 cup sugar until dissolved",
            "Combine lemon juice, simple syrup, and 4 cups cold water in pitcher",
            "Stir well and chill for at least 1 hour",
            "Serve over ice with lemon slices"
        ],
        dietary_info={
            "vegetarian": True,
            "vegan": True,
            "gluten_free": True
        }
    ),
    "garlic bread": Recipe(
        name="Garlic Bread",
        ingredients={
            "bread": "1 baguette",
            "garlic": "3 cloves",
            "butter": "1/2 cup",
            "parsley": "2 tbsp",
            "salt": "1/4 tsp"
        },
        steps=[
            "Preheat oven to 375°F (190°C)",
            "Melt 1/2 cup butter and mix with 3 minced garlic cloves",
            "Add 2 tbsp chopped parsley and 1/4 tsp salt",
            "Slice a baguette lengthwise and spread the garlic butter mixture",
            "Bake for 10-12 minutes until golden and crispy"
        ],
        dietary_info={
            "vegetarian": True,
            "vegan": False,
            "gluten_free": False
        }
    ),
}
