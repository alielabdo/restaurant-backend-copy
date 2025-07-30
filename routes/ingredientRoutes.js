const express = require('express');
const router = express.Router();
const Ingredient = require('../models/Ingredient');

// Create an ingredient
router.post('/', async (req, res) => {
  try {
    const { name, unit, currentStock, alertThreshold } = req.body;

    if (!name || !unit) {
      return res.status(400).json({ message: 'Name and unit are required' });
    }

    const ingredientExists = await Ingredient.findOne({ name });
    if (ingredientExists) {
      return res.status(400).json({ message: 'Ingredient already exists' });
    }

    const ingredient = new Ingredient({
      name,
      unit,
      currentStock: currentStock || 0,
      alertThreshold: alertThreshold || 0,
    });

    await ingredient.save();
    res.status(201).json(ingredient);
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

// Get all ingredients
router.get('/', async (req, res) => {
  try {
    const ingredients = await Ingredient.find({});
    res.json(ingredients);
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

// Get ingredient by id
router.get('/:id', async (req, res) => {
  try {
    const ingredient = await Ingredient.findById(req.params.id);
    if (!ingredient) {
      return res.status(404).json({ message: 'Ingredient not found' });
    }
    res.json(ingredient);
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

// Update ingredient by id
router.put('/:id', async (req, res) => {
  try {
    const { name, unit, currentStock, alertThreshold } = req.body;

    const ingredient = await Ingredient.findById(req.params.id);
    if (!ingredient) {
      return res.status(404).json({ message: 'Ingredient not found' });
    }

    // Update fields if provided
    if (name) ingredient.name = name;
    if (unit) ingredient.unit = unit;
    if (currentStock !== undefined) ingredient.currentStock = currentStock;
    if (alertThreshold !== undefined) ingredient.alertThreshold = alertThreshold;

    await ingredient.save();
    res.json(ingredient);
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

// Delete ingredient by id
router.delete('/:id', async (req, res) => {
  try {
    const ingredient = await Ingredient.findById(req.params.id);
    if (!ingredient) {
      return res.status(404).json({ message: 'Ingredient not found' });
    }

    await ingredient.remove();
    res.json({ message: 'Ingredient deleted' });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

module.exports = router;
