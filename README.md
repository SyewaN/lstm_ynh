# ESP32 Monitor - Kurulum ve Çalıştırma

Bu proje:
- ESP32'den Bluetooth ile gelen tuz/sıcaklık verisini alır
- MySQL'e kaydeder
- LSTM modeli ile tahmin üretir
- Dashboard üzerinden izleme ve yönetim sağlar

## Dizin Yapısı

```text
/var/www/esp32monitor/
  api/
    server.js
    package.json
    .env
    ecosystem.config.js
  lstm/
    app.py
    requirements.txt
    esp32-lstm.service
    models/
      model.h5
      scaler.pkl
  dashboard/
    index.html
```

## 1) Klasörleri Oluştur

```bash
mkdir -p /var/www/esp32monitor/{api,lstm/models,dashboard}
```

## 2) MySQL Veritabanı ve Tablolar

Önce MySQL'e gir:

```bash
mysql -u root -p
```

Ardından şu SQL komutlarını çalıştır:

```sql
CREATE DATABASE IF NOT EXISTS esp32monitor;
USE esp32monitor;

CREATE TABLE IF NOT EXISTS sensor_data (
  id INT AUTO_INCREMENT PRIMARY KEY,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
  salt FLOAT NOT NULL,
  sicaklik FLOAT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
  id INT AUTO_INCREMENT PRIMARY KEY,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
  predicted_salt FLOAT,
  predicted_sicaklik FLOAT
);
```

Not: `esp32user` kullanıcısını ve şifresini sunucunuzda oluşturup yetki verin.

## 3) Node.js API Kurulumu

```bash
cd /var/www/esp32monitor/api
npm install
pm2 start ecosystem.config.js
pm2 save
pm2 startup
```

`pm2 startup` çıktısında verilen komutu ayrıca çalıştırın.

## 4) Python LSTM Servisi Kurulumu

```bash
cd /var/www/esp32monitor/lstm
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate
sudo cp esp32-lstm.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable esp32-lstm
sudo systemctl start esp32-lstm
```

Servis kontrolü:

```bash
sudo systemctl status esp32-lstm
```

## 5) NGINX (YunoHost) Reverse Proxy Ayarı

```bash
sudo nano /etc/nginx/conf.d/whoogel.syewan.ynh.fr.conf
```

`server {}` bloğuna şu `location` bloklarını ekleyin:

```nginx
location /api {
    proxy_pass http://127.0.0.1:3001;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_read_timeout 300;
}

location /dashboard {
    alias /var/www/esp32monitor/dashboard;
    index index.html;
}
```

Sonra konfigürasyonu test edip yeniden yükleyin:

```bash
sudo nginx -t
sudo nginx -s reload
```

## 6) Test

```bash
curl https://whoogel.syewan.ynh.fr/api/health
curl https://whoogel.syewan.ynh.fr/dashboard
```

## Notlar

- API varsayılan port: `3001`
- LSTM servisi varsayılan port: `5001`
- Dashboard API çağrıları sabit olarak `https://whoogel.syewan.ynh.fr` adresine gider.
- Model dosyaları eğitimden sonra oluşur:
  - `/var/www/esp32monitor/lstm/models/model.h5`
  - `/var/www/esp32monitor/lstm/models/scaler.pkl`
