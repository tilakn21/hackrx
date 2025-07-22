#!/bin/bash
# EC2 Setup Script for HackRX LLM API
# Run this script on a fresh Ubuntu 22.04 EC2 instance

set -e

echo "🚀 Starting HackRX LLM API deployment setup..."

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
echo "🐍 Installing Python 3.11..."
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt install -y \
    nginx \
    git \
    curl \
    build-essential \
    htop \
    unzip \
    supervisor

# Install Node.js and PM2
echo "📊 Installing Node.js and PM2..."
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo npm install -g pm2

# Create application directory
echo "📁 Setting up application directory..."
sudo mkdir -p /opt/hackrx
sudo chown ubuntu:ubuntu /opt/hackrx
cd /opt/hackrx

# Clone repository
echo "📥 Cloning repository..."
if [ ! -d ".git" ]; then
    git clone https://github.com/tilakn21/hackrx.git .
else
    git pull origin master
fi

# Create virtual environment
echo "🌐 Creating Python virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Test installation
echo "🧪 Testing installation..."
python -c "import fastapi, uvicorn, sentence_transformers; print('✅ All packages installed successfully')"

# Create systemd service
echo "⚙️ Creating systemd service..."
sudo tee /etc/systemd/system/hackrx.service > /dev/null <<EOF
[Unit]
Description=HackRX LLM API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/hackrx
Environment=PATH=/opt/hackrx/venv/bin
Environment=PYTHONPATH=/opt/hackrx
Environment=PORT=8000
ExecStart=/opt/hackrx/venv/bin/python main.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable hackrx
sudo systemctl start hackrx

# Configure Nginx
echo "🌐 Configuring Nginx..."
sudo tee /etc/nginx/sites-available/hackrx > /dev/null <<EOF
server {
    listen 80;
    server_name _;

    # Rate limiting
    limit_req_zone \$binary_remote_addr zone=api:10m rate=10r/s;

    location / {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        access_log off;
    }
}
EOF

# Enable Nginx site
sudo ln -sf /etc/nginx/sites-available/hackrx /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx

# Configure firewall
echo "🔒 Configuring firewall..."
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'
sudo ufw --force enable

# Create monitoring script
echo "📊 Creating monitoring script..."
tee /opt/hackrx/monitor.sh > /dev/null <<EOF
#!/bin/bash
# Monitor HackRX API health

check_service() {
    if systemctl is-active --quiet hackrx; then
        echo "✅ HackRX service is running"
    else
        echo "❌ HackRX service is not running"
        sudo systemctl restart hackrx
    fi
}

check_health() {
    if curl -s http://localhost:8000/health > /dev/null; then
        echo "✅ Health endpoint responding"
    else
        echo "❌ Health endpoint not responding"
    fi
}

echo "=== HackRX Monitoring $(date) ==="
check_service
check_health
echo "================================"
EOF

chmod +x /opt/hackrx/monitor.sh

# Add cron job for monitoring
echo "⏰ Setting up monitoring cron job..."
(crontab -l 2>/dev/null; echo "*/5 * * * * /opt/hackrx/monitor.sh >> /var/log/hackrx-monitor.log 2>&1") | crontab -

# Final status check
echo "🏁 Final status check..."
sleep 5
sudo systemctl status hackrx --no-pager
curl -s http://localhost:8000/health || echo "Health check will be available once the service fully starts"

echo "✅ Setup complete!"
echo ""
echo "📋 Important Information:"
echo "- Application is running on port 8000"
echo "- Nginx is proxying traffic on port 80"
echo "- Service name: hackrx"
echo "- Logs: sudo journalctl -u hackrx -f"
echo "- Config: /etc/systemd/system/hackrx.service"
echo "- Monitor: /opt/hackrx/monitor.sh"
echo ""
echo "🌐 Test your deployment:"
echo "curl http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)/health"
echo ""
echo "🛠️ Useful commands:"
echo "sudo systemctl status hackrx    # Check service status"
echo "sudo systemctl restart hackrx   # Restart service"
echo "sudo journalctl -u hackrx -f    # View logs"
echo "pm2 monit                       # Monitor processes"
