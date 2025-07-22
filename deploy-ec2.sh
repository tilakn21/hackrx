#!/bin/bash
# Quick deployment script for AWS EC2

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 HackRX LLM API - AWS EC2 Deployment${NC}"
echo "=================================================="

# Check if running on EC2
if ! curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/instance-id > /dev/null; then
    echo -e "${RED}❌ This script should be run on an AWS EC2 instance${NC}"
    exit 1
fi

# Get instance info
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
PRIVATE_IP=$(curl -s http://169.254.169.254/latest/meta-data/local-ipv4)

echo -e "${YELLOW}Instance ID:${NC} $INSTANCE_ID"
echo -e "${YELLOW}Public IP:${NC} $PUBLIC_IP"
echo -e "${YELLOW}Private IP:${NC} $PRIVATE_IP"
echo ""

# Run setup
if [ -f "setup-ec2.sh" ]; then
    echo -e "${GREEN}Running EC2 setup script...${NC}"
    chmod +x setup-ec2.sh
    ./setup-ec2.sh
else
    echo -e "${RED}❌ setup-ec2.sh not found${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
echo ""
echo -e "${YELLOW}🌐 Your API is available at:${NC}"
echo "   Health Check: http://$PUBLIC_IP/health"
echo "   API Endpoint: http://$PUBLIC_IP/api/v1/hackrx/run"
echo ""
echo -e "${YELLOW}📊 Monitoring:${NC}"
echo "   Service Status: sudo systemctl status hackrx"
echo "   Logs: sudo journalctl -u hackrx -f"
echo "   Nginx Status: sudo systemctl status nginx"
echo ""
echo -e "${YELLOW}🔧 Management Commands:${NC}"
echo "   Restart API: sudo systemctl restart hackrx"
echo "   Update Code: cd /opt/hackrx && git pull && sudo systemctl restart hackrx"
echo "   View Logs: sudo journalctl -u hackrx -n 50"
