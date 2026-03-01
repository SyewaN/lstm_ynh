-- Bu dosya MySQL veritabani semasini manuel kurulum icin icerir.

CREATE TABLE IF NOT EXISTS sensor_data (
  id INT AUTO_INCREMENT PRIMARY KEY,
  timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  salt FLOAT NOT NULL,
  sicaklik FLOAT NOT NULL,
  INDEX idx_sensor_timestamp (timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS predictions (
  id INT AUTO_INCREMENT PRIMARY KEY,
  timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  predicted_salt FLOAT NOT NULL,
  predicted_sicaklik FLOAT NOT NULL,
  model_version VARCHAR(100) NOT NULL,
  INDEX idx_predictions_timestamp (timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
