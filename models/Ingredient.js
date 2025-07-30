const mongoose = require('mongoose');

const ingredientSchema = new mongoose.Schema(
  {
    name: {
      type: String,
      required: [true, 'Ingredient name is required'],
      unique: true,
      trim: true,
    },
    unit: {
      type: String,
      enum: ['g', 'kg', 'ml', 'l', 'piece', 'unit'],
      required: [true, 'Unit is required'],
    },
    currentStock: {
      type: Number,
      default: 0, // Can represent quantity in kg, liters, or pieces
    },
    alertThreshold: {
      type: Number,
      default: 0, // When stock drops below this, notify
    },
  },
  {
    timestamps: true,
  }
);

const Ingredient = mongoose.model('Ingredient', ingredientSchema);

module.exports = Ingredient;
