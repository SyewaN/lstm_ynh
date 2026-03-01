const express = require('express');
const cors = require('cors');
const mysql = require('mysql2/promise');
const dotenv = require('dotenv');
const axios = require('axios');

dotenv.config();

const app = express();
const PORT = Number(process.env.PORT || 3001);
const LSTM_URL = process.env.LSTM_URL || 'http://127.0.0.1:5001';

// Tüm originlere CORS izni ver.
app.use(cors());
// JSON body parse et.
app.use(express.json());

// MySQL bağlantı havuzu.
const pool = mysql.createPool({
  host: process.env.DB_HOST,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0
});

function toCsv(rows) {
  // CSV alanlarında çift tırnak kaçışını güvenli şekilde yap.
  const header = 'id,timestamp,salt,sicaklik';
  const lines = rows.map((row) => {
    const fields = [row.id, row.timestamp, row.salt, row.sicaklik].map((value) => {
      const text = String(value ?? '');
      return `"${text.replace(/"/g, '""')}"`;
    });
    return fields.join(',');
  });
  return [header, ...lines].join('\n');
}

app.post('/api/data', async (req, res) => {
  try {
    const { salt, sicaklik } = req.body;

    if (typeof salt !== 'number' || typeof sicaklik !== 'number') {
      return res.status(400).json({ error: 'Geçersiz veri. salt ve sicaklik sayısal olmalı.' });
    }

    const [result] = await pool.execute(
      'INSERT INTO sensor_data (salt, sicaklik) VALUES (?, ?)',
      [salt, sicaklik]
    );

    return res.json({ success: true, id: result.insertId });
  } catch (error) {
    return res.status(500).json({ error: 'Veri kaydı sırasında hata oluştu', detail: error.message });
  }
});

app.get('/api/data', async (_req, res) => {
  try {
    const [rows] = await pool.execute(
      'SELECT id, timestamp, salt, sicaklik FROM sensor_data ORDER BY timestamp DESC LIMIT 100'
    );
    return res.json(rows);
  } catch (error) {
    return res.status(500).json({ error: 'Veriler alınamadı', detail: error.message });
  }
});

app.get('/api/stats', async (_req, res) => {
  try {
    const [statsRows] = await pool.execute(`
      SELECT
        COUNT(*) AS total,
        AVG(salt) AS avg_salt,
        AVG(sicaklik) AS avg_sicaklik,
        MIN(salt) AS min_salt,
        MAX(salt) AS max_salt,
        MIN(sicaklik) AS min_sicaklik,
        MAX(sicaklik) AS max_sicaklik
      FROM sensor_data
    `);

    const [lastRows] = await pool.execute(
      'SELECT salt AS last_salt, sicaklik AS last_sicaklik FROM sensor_data ORDER BY timestamp DESC LIMIT 1'
    );

    const stats = statsRows[0] || {};
    const last = lastRows[0] || { last_salt: null, last_sicaklik: null };

    return res.json({
      total: Number(stats.total || 0),
      last_salt: last.last_salt,
      last_sicaklik: last.last_sicaklik,
      avg_salt: stats.avg_salt,
      avg_sicaklik: stats.avg_sicaklik,
      min_salt: stats.min_salt,
      max_salt: stats.max_salt,
      min_sicaklik: stats.min_sicaklik,
      max_sicaklik: stats.max_sicaklik
    });
  } catch (error) {
    return res.status(500).json({ error: 'İstatistikler alınamadı', detail: error.message });
  }
});

app.get('/api/predict', async (_req, res) => {
  try {
    const response = await axios.get(`${LSTM_URL}/predict`, { timeout: 30000 });
    return res.json(response.data);
  } catch (error) {
    const detail = error.response?.data || error.message;
    return res.status(503).json({ error: 'ML servisi hazır değil', detail });
  }
});

app.get('/api/train', async (_req, res) => {
  try {
    const response = await axios.get(`${LSTM_URL}/train`, { timeout: 300000 });
    return res.json(response.data);
  } catch (error) {
    const detail = error.response?.data || error.message;
    return res.status(500).json({ error: 'Model eğitimi başlatılamadı', detail });
  }
});

app.get('/api/health', async (_req, res) => {
  let dbStatus = 'ok';
  let lstmStatus = 'ok';

  try {
    await pool.query('SELECT 1');
  } catch (_error) {
    dbStatus = 'err';
  }

  try {
    await axios.get(`${LSTM_URL}/health`, { timeout: 5000 });
  } catch (_error) {
    lstmStatus = 'err';
  }

  return res.json({ api: 'ok', db: dbStatus, lstm: lstmStatus });
});

app.get('/api/export', async (_req, res) => {
  try {
    const [rows] = await pool.execute(
      'SELECT id, timestamp, salt, sicaklik FROM sensor_data ORDER BY timestamp ASC'
    );

    const csvData = toCsv(rows);
    res.setHeader('Content-Type', 'text/csv; charset=utf-8');
    res.setHeader('Content-Disposition', 'attachment; filename="sensor_data.csv"');
    return res.status(200).send(csvData);
  } catch (error) {
    return res.status(500).json({ error: 'CSV dışa aktarma başarısız', detail: error.message });
  }
});

// Tanımsız endpoint'ler için standart JSON hata çıktısı.
app.use((_req, res) => {
  res.status(404).json({ error: 'Endpoint bulunamadı' });
});

app.listen(PORT, () => {
  console.log(`ESP32 API dinleniyor: ${PORT}`);
});
