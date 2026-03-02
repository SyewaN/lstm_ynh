#!/bin/bash
set -e

echo "=== Klasörler oluşturuluyor ==="
mkdir -p /var/www/esp32monitor/{api,lstm/models,dashboard}

echo "=== Veritabanı kuruluyor ==="
bash setup_db.sh

echo "=== Node.js API kuruluyor ==="
cd /var/www/esp32monitor/api
npm install
pm2 start ecosystem.config.js
pm2 save

echo "=== Python LSTM kuruluyor ==="
cd /var/www/esp32monitor/lstm
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate
cp esp32-lstm.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable esp32-lstm
systemctl start esp32-lstm

echo "=== Test ==="
sleep 3
curl -s http://127.0.0.1:3001/api/health
curl -s http://127.0.0.1:5001/health

echo "=== TAMAMLANDI ==="
echo "Şimdi YunoHost panelinden Redirect uygulaması kurun:"
echo "  Path: /api  → Target: http://127.0.0.1:3001"
echo "  Path: /lstm → Target: http://127.0.0.1:5001"
