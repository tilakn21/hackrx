module.exports = {
  apps: [{
    name: 'hackrx-api',
    script: './venv/bin/python',
    args: 'main.py',
    cwd: '/opt/hackrx',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
    env: {
      NODE_ENV: 'production',
      PORT: 8000,
      PYTHONPATH: '/opt/hackrx',
      PYTHONUNBUFFERED: '1'
    },
    error_file: '/opt/hackrx/logs/err.log',
    out_file: '/opt/hackrx/logs/out.log',
    log_file: '/opt/hackrx/logs/combined.log',
    time: true
  }]
};
