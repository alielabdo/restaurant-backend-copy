const express = require('express');
const router = express.Router();
const Item = require('../models/Item');
const Ingredient = require('../models/Ingredient');

const {protect} = require('../middlewares/authMiddleware'); // If you want routes protected

// Create new item
router.post('/', protect, async (req, res) => {
  try {
    const { name, category, price, ingredients } = req.body;

    if (!name || !category || !price) {
      return res.status(400).json({ message: 'Name, category and price are required' });
    }

    const existingItem = await Item.findOne({ name });
    if (existingItem) {
      return res.status(400).json({ message: 'Item with this name already exists' });
    }

    // Validate each ingredient
    if (ingredients && ingredients.length > 0) {
      for (const ing of ingredients) {
        if (!ing.ingredient || !ing.quantity || !ing.unit) {
          return res.status(400).json({ message: 'Each ingredient must have ingredient ID, quantity and unit' });
        }
      }
    }

    const newItem = new Item({
      name,
      category,
      price,
      ingredients,
    });

    await newItem.save();

    res.status(201).json(newItem);
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

// Get all items
router.get('/', protect, async (req, res) => {
  try {
    const items = await Item.find({}).populate('ingredients.ingredient', 'name unit');
    res.json(items);
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

// Get one item by id
router.get('/:id', protect, async (req, res) => {
  try {
    const item = await Item.findById(req.params.id).populate('ingredients.ingredient', 'name unit');
    if (!item) {
      return res.status(404).json({ message: 'Item not found' });
    }
    res.json(item);
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

// Update item
router.put('/:id', protect, async (req, res) => {
  try {
    const { name, category, price, ingredients } = req.body;

    const item = await Item.findById(req.params.id);
    if (!item) {
      return res.status(404).json({ message: 'Item not found' });
    }

    // Update fields if provided
    if (name) item.name = name;
    if (category) item.category = category;
    if (price) item.price = price;
    if (ingredients) item.ingredients = ingredients;

    await item.save();

    res.json(item);
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

// Delete item
router.delete('/:id', protect, async (req, res) => {
  try {
    const item = await Item.findById(req.params.id);
    if (!item) {
      return res.status(404).json({ message: 'Item not found' });
    }

    await item.remove();
    res.json({ message: 'Item removed' });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

module.exports = router;
