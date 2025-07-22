# AWS EC2 Deployment Guide for HackRX LLM API

## Prerequisites
1. AWS Account
2. AWS CLI installed and configured
3. SSH key pair for EC2 access
4. Domain name (optional, for custom domain)

## Step 1: Launch EC2 Instance

### Instance Configuration:
- **Instance Type**: t3.medium (2 vCPU, 4GB RAM) - Free tier: t2.micro
- **AMI**: Ubuntu 22.04 LTS
- **Storage**: 20GB gp3 SSD
- **Security Group**: Allow HTTP (80), HTTPS (443), SSH (22), Custom (8000)

### AWS CLI Commands:
```bash
# Create security group
aws ec2 create-security-group \
    --group-name hackrx-sg \
    --description "Security group for HackRX LLM API"

# Add inbound rules
aws ec2 authorize-security-group-ingress \
    --group-name hackrx-sg \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-name hackrx-sg \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-name hackrx-sg \
    --protocol tcp \
    --port 443 \
    --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-name hackrx-sg \
    --protocol tcp \
    --port 8000 \
    --cidr 0.0.0.0/0

# Launch instance
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --count 1 \
    --instance-type t3.medium \
    --key-name your-key-pair \
    --security-groups hackrx-sg \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=hackrx-llm-api}]'
```

## Step 2: Connect to Instance

```bash
# SSH into your instance
ssh -i your-key.pem ubuntu@your-instance-ip
```

## Step 3: Server Setup

Run the setup script (see setup.sh) or execute manually:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Install system dependencies
sudo apt install -y nginx git curl build-essential

# Install Docker (optional)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Node.js and PM2 (for process management)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo npm install -g pm2
```

## Step 4: Deploy Application

```bash
# Clone repository
git clone https://github.com/tilakn21/hackrx.git
cd hackrx

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Test application
python main.py &
curl http://localhost:10000/health
```

## Step 5: Configure Process Manager

Use PM2 to manage the application process:

```bash
# Create PM2 ecosystem file (see ecosystem.config.js)
pm2 start ecosystem.config.js
pm2 save
pm2 startup
```

## Step 6: Configure Nginx (Reverse Proxy)

```bash
# Create Nginx configuration (see nginx.conf)
sudo cp nginx.conf /etc/nginx/sites-available/hackrx
sudo ln -s /etc/nginx/sites-available/hackrx /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```

## Step 7: SSL Certificate (Optional)

```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com
```

## Step 8: Configure Firewall

```bash
# Setup UFW
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'
sudo ufw enable
```

## Monitoring and Maintenance

### Check Application Status:
```bash
pm2 status
pm2 logs hackrx-api
sudo systemctl status nginx
```

### Update Application:
```bash
cd /home/ubuntu/hackrx
git pull origin master
source venv/bin/activate
pip install -r requirements.txt
pm2 restart hackrx-api
```

### Backup and Recovery:
```bash
# Create snapshot of EBS volume
aws ec2 create-snapshot --volume-id vol-xxxxxxxx --description "HackRX backup"
```

## Cost Optimization

### For Free Tier:
- Use t2.micro instance
- 8GB storage
- Stop instance when not in use

### For Production:
- Use t3.medium or larger
- Set up Auto Scaling Group
- Use Application Load Balancer
- Set up CloudWatch monitoring

## Security Best Practices

1. **Regular Updates**: Keep system and packages updated
2. **SSH Key Security**: Use strong SSH keys, disable password auth
3. **Firewall**: Only open necessary ports
4. **SSL/TLS**: Use HTTPS in production
5. **Environment Variables**: Store sensitive data securely
6. **Monitoring**: Set up CloudWatch logs and metrics

## Troubleshooting

### Common Issues:
1. **Port conflicts**: Check if port 10000 is available
2. **Memory issues**: Monitor with `htop`, upgrade instance if needed
3. **Disk space**: Use `df -h` to check disk usage
4. **Process crashes**: Check PM2 logs with `pm2 logs`

### Logs Location:
- Application logs: `pm2 logs hackrx-api`
- Nginx logs: `/var/log/nginx/`
- System logs: `/var/log/syslog`
