/*
Bu dosya ESP32 verilerini alan, MySQL'e kaydeden ve ML servisinden tahmin alan Node.js API sunucusudur.
*/

const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const mysql = require('mysql2/promise');
const axios = require('axios');
const path = require('path');

dotenv.config();

const app = express();
const port = Number(process.env.PORT || 3000);
const mlServiceUrl = process.env.ML_SERVICE_URL || 'http://127.0.0.1:5000';
const apiKey = process.env.API_KEY;
const githubUsername = process.env.GITHUB_USERNAME || 'kullanici';

if (!apiKey) {
  console.error('HATA: API_KEY tanimli degil. Lutfen .env dosyasini kontrol edin.');
  process.exit(1);
}

const allowedOriginRegex = new RegExp(`^https://${githubUsername}\\.github\\.io$`);

app.use(cors({
  origin: (origin, callback) => {
    if (!origin) {
      callback(null, true);
      return;
    }

    const isGithubPages = allowedOriginRegex.test(origin);
    const isLocalhost = /^http:\/\/localhost(:\d+)?$/.test(origin);

    if (isGithubPages || isLocalhost) {
      callback(null, true);
      return;
    }

    callback(new Error('CORS hatasi: Bu origin icin erisim izni yok.'));
  }
}));

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

const dbPool = mysql.createPool({
  host: process.env.DB_HOST || '127.0.0.1',
  port: Number(process.env.DB_PORT || 3306),
  user: process.env.DB_USER,
  password: process.env.DB_PASS,
  database: process.env.DB_NAME,
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0
});

async function createTablesIfNotExists() {
  await dbPool.query(`
    CREATE TABLE IF NOT EXISTS sensor_data (
      id INT AUTO_INCREMENT PRIMARY KEY,
      timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
      salt FLOAT NOT NULL,
      sicaklik FLOAT NOT NULL,
      INDEX idx_sensor_timestamp (timestamp)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
  `);

  await dbPool.query(`
    CREATE TABLE IF NOT EXISTS predictions (
      id INT AUTO_INCREMENT PRIMARY KEY,
      timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
      predicted_salt FLOAT NOT NULL,
      predicted_sicaklik FLOAT NOT NULL,
      model_version VARCHAR(100) NOT NULL,
      INDEX idx_predictions_timestamp (timestamp)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
  `);
}

function validateApiKey(req, res, next) {
  const incomingApiKey = req.header('x-api-key');
  if (!incomingApiKey || incomingApiKey !== apiKey) {
    return res.status(401).json({ error: 'Gecersiz veya eksik API key' });
  }
  return next();
}

app.get('/health', (_req, res) => {
  res.json({ status: 'ok', service: 'node-api' });
});

app.get('/', (_req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.post('/api/data', validateApiKey, async (req, res) => {
  try {
    const salt = Number(req.body.salt);
    const sicaklik = Number(req.body.sicaklik);

    if (!Number.isFinite(salt) || !Number.isFinite(sicaklik)) {
      return res.status(400).json({ error: 'salt ve sicaklik sayisal olmalidir' });
    }

    const [result] = await dbPool.query(
      'INSERT INTO sensor_data (salt, sicaklik) VALUES (?, ?)',
      [salt, sicaklik]
    );

    return res.status(201).json({
      message: 'Veri basariyla kaydedildi',
      id: result.insertId,
      salt,
      sicaklik
    });
  } catch (error) {
    console.error('POST /api/data hatasi:', error.message);
    return res.status(500).json({ error: 'Veri kaydedilirken bir hata olustu' });
  }
});

app.get('/api/data', async (_req, res) => {
  try {
    const [rows] = await dbPool.query(`
      SELECT id, timestamp, salt, sicaklik
      FROM sensor_data
      ORDER BY timestamp DESC
      LIMIT 1000
    `);

    return res.json(rows);
  } catch (error) {
    console.error('GET /api/data hatasi:', error.message);
    return res.status(500).json({ error: 'Veriler alinirken bir hata olustu' });
  }
});

app.get('/api/data/stats', async (_req, res) => {
  try {
    const [statsRows] = await dbPool.query(`
      SELECT
        AVG(salt) AS avg_salt,
        MIN(salt) AS min_salt,
        MAX(salt) AS max_salt,
        AVG(sicaklik) AS avg_sicaklik,
        MIN(sicaklik) AS min_sicaklik,
        MAX(sicaklik) AS max_sicaklik,
        COUNT(*) AS total_count
      FROM sensor_data
    `);

    const [lastRows] = await dbPool.query(`
      SELECT id, timestamp, salt, sicaklik
      FROM sensor_data
      ORDER BY timestamp DESC
      LIMIT 1
    `);

    return res.json({
      statistics: statsRows[0],
      last_value: lastRows[0] || null
    });
  } catch (error) {
    console.error('GET /api/data/stats hatasi:', error.message);
    return res.status(500).json({ error: 'Istatistikler alinirken bir hata olustu' });
  }
});

app.get('/api/dashboard', async (req, res) => {
  try {
    const limitFromQuery = Number(req.query.limit || 100);
    const limit = Number.isFinite(limitFromQuery)
      ? Math.min(Math.max(Math.trunc(limitFromQuery), 1), 1000)
      : 100;

    const [dataRows] = await dbPool.query(`
      SELECT id, timestamp, salt, sicaklik
      FROM sensor_data
      ORDER BY timestamp DESC
      LIMIT ?
    `, [limit]);

    const [statsRows] = await dbPool.query(`
      SELECT
        AVG(salt) AS avg_salt,
        MIN(salt) AS min_salt,
        MAX(salt) AS max_salt,
        AVG(sicaklik) AS avg_sicaklik,
        MIN(sicaklik) AS min_sicaklik,
        MAX(sicaklik) AS max_sicaklik,
        COUNT(*) AS total_count
      FROM sensor_data
    `);

    const [lastPredictionRows] = await dbPool.query(`
      SELECT id, timestamp, predicted_salt, predicted_sicaklik, model_version
      FROM predictions
      ORDER BY timestamp DESC
      LIMIT 1
    `);

    return res.json({
      data: dataRows,
      statistics: statsRows[0],
      last_value: dataRows[0] || null,
      last_prediction: lastPredictionRows[0] || null
    });
  } catch (error) {
    console.error('GET /api/dashboard hatasi:', error.message);
    return res.status(500).json({ error: 'Dashboard verisi alinirken bir hata olustu' });
  }
});

app.get('/api/train', async (_req, res) => {
  try {
    const response = await axios.get(`${mlServiceUrl}/train`, {
      timeout: 120000
    });
    return res.status(response.status).json(response.data);
  } catch (error) {
    console.error('GET /api/train hatasi:', error.message);

    if (error.response) {
      return res.status(error.response.status).json({
        error: 'ML egitim servisi hata dondu',
        details: error.response.data
      });
    }

    return res.status(502).json({ error: 'ML egitim servisine ulasilamadi' });
  }
});

app.get('/api/predict', validateApiKey, async (_req, res) => {
  try {
    const response = await axios.get(`${mlServiceUrl}/predict`, {
      timeout: 30000
    });

    const { predicted_salt, predicted_sicaklik, model_version } = response.data;

    if (!Number.isFinite(Number(predicted_salt)) || !Number.isFinite(Number(predicted_sicaklik))) {
      return res.status(502).json({ error: 'ML servisi gecersiz tahmin verisi dondu' });
    }

    await dbPool.query(
      'INSERT INTO predictions (predicted_salt, predicted_sicaklik, model_version) VALUES (?, ?, ?)',
      [Number(predicted_salt), Number(predicted_sicaklik), model_version || 'v1']
    );

    return res.json({
      predicted_salt: Number(predicted_salt),
      predicted_sicaklik: Number(predicted_sicaklik),
      model_version: model_version || 'v1'
    });
  } catch (error) {
    console.error('GET /api/predict hatasi:', error.message);

    if (error.response) {
      return res.status(error.response.status).json({
        error: 'ML servisi hata dondu',
        details: error.response.data
      });
    }

    return res.status(502).json({ error: 'ML servisine ulasilamadi' });
  }
});

async function startServer() {
  try {
    await createTablesIfNotExists();
    app.listen(port, () => {
      console.log(`Node API calisiyor: http://0.0.0.0:${port}`);
    });
  } catch (error) {
    console.error('Sunucu baslatilamadi:', error.message);
    process.exit(1);
  }
}

startServer();
