const express = require('express');
const cors = require('cors');
require('dotenv').config();
const connectDB = require('./config/db');



const authRoutes = require('./routes/authRoutes');
const protectedRoutes = require('./routes/protectedRoutes');
const ingredientRoutes = require('./routes/ingredientRoutes');
const itemRoutes = require('./routes/itemRoutes');



const app = express();


connectDB();


app.use(cors());
app.use(express.json());
app.use('/api/ingredients', ingredientRoutes);
app.use('/api/ingredients', ingredientRoutes);
app.use('/api/items', itemRoutes);
app.use('/api/auth', authRoutes);
app.use('/api/protected', protectedRoutes);
app.get('/', (req, res) => {
  res.send('Restaurant API is running ');
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(` Server running at http://localhost:${PORT}`);
});
