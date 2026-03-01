module.exports = {
  apps: [
    {
      name: 'esp32-api',
      script: 'server.js',
      cwd: '/var/www/esp32monitor/api',
      instances: 1,
      autorestart: true,
      watch: false,
      env: {
        NODE_ENV: 'production'
      }
    }
  ]
};
