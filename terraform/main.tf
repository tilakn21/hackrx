# Terraform configuration for AWS EC2 deployment
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.medium"  # Use t2.micro for free tier
}

variable "key_name" {
  description = "EC2 Key Pair name"
  type        = string
}

# Data sources
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Security Group
resource "aws_security_group" "hackrx_sg" {
  name_prefix = "hackrx-sg"
  description = "Security group for HackRX LLM API"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "hackrx-security-group"
  }
}

# EC2 Instance
resource "aws_instance" "hackrx_api" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = var.instance_type
  key_name      = var.key_name

  vpc_security_group_ids = [aws_security_group.hackrx_sg.id]

  root_block_device {
    volume_type = "gp3"
    volume_size = 20
    encrypted   = true
  }

  user_data = file("${path.module}/user-data.sh")

  tags = {
    Name = "hackrx-llm-api"
    Environment = "production"
  }
}

# Elastic IP
resource "aws_eip" "hackrx_eip" {
  instance = aws_instance.hackrx_api.id
  domain   = "vpc"

  tags = {
    Name = "hackrx-eip"
  }
}

# Outputs
output "instance_id" {
  description = "ID of the EC2 instance"
  value       = aws_instance.hackrx_api.id
}

output "instance_public_ip" {
  description = "Public IP address of the EC2 instance"
  value       = aws_eip.hackrx_eip.public_ip
}

output "instance_public_dns" {
  description = "Public DNS name of the EC2 instance"
  value       = aws_instance.hackrx_api.public_dns
}

output "health_check_url" {
  description = "Health check URL"
  value       = "http://${aws_eip.hackrx_eip.public_ip}/health"
}

output "api_url" {
  description = "API base URL"
  value       = "http://${aws_eip.hackrx_eip.public_ip}"
}
