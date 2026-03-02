#!/bin/bash
# MySQL root şifresi sorar, DB ve kullanıcı oluşturur
read -sp "MySQL root şifresi: " MYSQL_ROOT_PASS
echo

mysql -u root -p"$MYSQL_ROOT_PASS" <<EOF_SQL
CREATE DATABASE IF NOT EXISTS esp32monitor CHARACTER SET utf8mb4;
CREATE USER IF NOT EXISTS 'esp32user'@'localhost' IDENTIFIED BY 'Esp32Pass123!';
GRANT ALL PRIVILEGES ON esp32monitor.* TO 'esp32user'@'localhost';
FLUSH PRIVILEGES;
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
EOF_SQL

echo "Veritabanı hazır!"
