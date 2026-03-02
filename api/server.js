require('dotenv').config();

const express = require('express');
const cors = require('cors');
const mysql = require('mysql2/promise');
const axios = require('axios');

const app = express();
const PORT = Number(process.env.PORT) || 3001;
const LSTM_URL = process.env.LSTM_URL || 'http://127.0.0.1:5001';

app.use(cors());
app.use(express.json());

// DB havuzu uygulama ayakta kalsin diye try/catch ile baslatilir.
let pool = null;

function createPool() {
  return mysql.createPool({
    host: process.env.DB_HOST || 'localhost',
    user: process.env.DB_USER || 'esp32user',
    password: process.env.DB_PASSWORD || '',
    database: process.env.DB_NAME || 'esp32monitor',
    waitForConnections: true,
    connectionLimit: 10,
    queueLimit: 0
  });
}

async function initDb() {
  try {
    pool = createPool();
    await pool.query('SELECT 1');
    console.log('MySQL baglantisi basarili.');
  } catch (err) {
    console.error('MySQL baglantisi basarisiz, API calismaya devam edecek:', err.message);
    pool = null;
  }
}

async function query(sql, params = []) {
  if (!pool) {
    pool = createPool();
  }
  return pool.query(sql, params);
}

app.post('/api/data', async (req, res) => {
  try {
    const { salt, sicaklik } = req.body || {};

    if (typeof salt !== 'number' || typeof sicaklik !== 'number') {
      return res.status(400).json({ error: 'Geçersiz veri. salt ve sicaklik sayi olmalidir.' });
    }

    const [result] = await query('INSERT INTO sensor_data (salt, sicaklik) VALUES (?, ?)', [salt, sicaklik]);
    return res.json({ success: true, id: result.insertId });
  } catch (err) {
    console.error('/api/data POST hatasi:', err.message);
    return res.status(500).json({ error: 'Veri kaydedilemedi' });
  }
});

app.get('/api/data', async (_req, res) => {
  try {
    const [rows] = await query(
      'SELECT id, timestamp, salt, sicaklik FROM sensor_data ORDER BY timestamp DESC LIMIT 100'
    );
    return res.json(rows);
  } catch (err) {
    console.error('/api/data GET hatasi:', err.message);
    return res.status(500).json({ error: 'Veriler alinamadi' });
  }
});

app.get('/api/stats', async (_req, res) => {
  try {
    const [totalRows] = await query('SELECT COUNT(*) AS total FROM sensor_data');
    const [lastRows] = await query(
      'SELECT salt, sicaklik FROM sensor_data ORDER BY timestamp DESC LIMIT 1'
    );
    const [aggRows] = await query(
      `SELECT
        AVG(salt) AS avg_salt,
        AVG(sicaklik) AS avg_sicaklik,
        MIN(salt) AS min_salt,
        MAX(salt) AS max_salt,
        MIN(sicaklik) AS min_sicaklik,
        MAX(sicaklik) AS max_sicaklik
      FROM sensor_data`
    );

    const total = totalRows?.[0]?.total || 0;
    const last = lastRows?.[0] || {};
    const agg = aggRows?.[0] || {};

    return res.json({
      total,
      last_salt: last.salt ?? null,
      last_sicaklik: last.sicaklik ?? null,
      avg_salt: agg.avg_salt ?? null,
      avg_sicaklik: agg.avg_sicaklik ?? null,
      min_salt: agg.min_salt ?? null,
      max_salt: agg.max_salt ?? null,
      min_sicaklik: agg.min_sicaklik ?? null,
      max_sicaklik: agg.max_sicaklik ?? null
    });
  } catch (err) {
    console.error('/api/stats hatasi:', err.message);
    return res.status(500).json({ error: 'Istatistikler alinamadi' });
  }
});

app.get('/api/predict', async (_req, res) => {
  try {
    const response = await axios.get(`${LSTM_URL}/predict`, { timeout: 15000 });
    return res.json(response.data);
  } catch (err) {
    console.error('/api/predict hatasi:', err.message);
    return res.status(503).json({ error: 'ML servisi hazır değil' });
  }
});

app.get('/api/train', async (_req, res) => {
  try {
    const response = await axios.get(`${LSTM_URL}/train`, { timeout: 10 * 60 * 1000 });
    return res.json(response.data);
  } catch (err) {
    console.error('/api/train hatasi:', err.message);
    return res.status(500).json({ error: 'Model egitimi baslatilamadi' });
  }
});

app.get('/api/health', async (_req, res) => {
  let dbStatus = 'err';
  let lstmStatus = 'err';

  try {
    await query('SELECT 1');
    dbStatus = 'ok';
  } catch (err) {
    console.error('/api/health DB hatasi:', err.message);
  }

  try {
    await axios.get(`${LSTM_URL}/health`, { timeout: 5000 });
    lstmStatus = 'ok';
  } catch (err) {
    console.error('/api/health LSTM hatasi:', err.message);
  }

  return res.json({ api: 'ok', db: dbStatus, lstm: lstmStatus });
});

app.get('/api/export', async (_req, res) => {
  try {
    const [rows] = await query(
      'SELECT id, timestamp, salt, sicaklik FROM sensor_data ORDER BY timestamp DESC'
    );

    const header = 'id,timestamp,salt,sicaklik';
    const lines = rows.map((row) => {
      const timestamp = new Date(row.timestamp).toISOString().replace('T', ' ').slice(0, 19);
      return `${row.id},${timestamp},${row.salt},${row.sicaklik}`;
    });
    const csv = [header, ...lines].join('\n');

    res.setHeader('Content-Type', 'text/csv; charset=utf-8');
    res.setHeader('Content-Disposition', 'attachment; filename="sensor_data.csv"');
    return res.send(csv);
  } catch (err) {
    console.error('/api/export hatasi:', err.message);
    return res.status(500).json({ error: 'CSV olusturulamadi' });
  }
});

app.use((err, _req, res, _next) => {
  console.error('Beklenmeyen API hatasi:', err);
  return res.status(500).json({ error: 'Sunucu hatasi' });
});

app.listen(PORT, async () => {
  await initDb();
  console.log(`API ${PORT} portunda calisiyor.`);
});
