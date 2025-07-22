#!/bin/bash
# User data script for EC2 instance initialization

# Log everything
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

# Update system
apt-get update
apt-get upgrade -y

# Install git
apt-get install -y git

# Clone repository and run setup
cd /tmp
git clone https://github.com/tilakn21/hackrx.git
cd hackrx
chmod +x setup-ec2.sh
./setup-ec2.sh

echo "User data script completed" >> /var/log/user-data.log
