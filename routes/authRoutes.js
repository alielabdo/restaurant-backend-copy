const express = require('express');
const { createUser, loginUser } = require('../controllers/authController');

const router = express.Router();


router.post('/create', createUser);

router.post('/login', loginUser);

module.exports = router;
