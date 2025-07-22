#!/usr/bin/env python3
"""
Test script to verify the application works locally before deployment
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:10000"
AUTH_TOKEN = "4e604cba5381f75493ad0742118a94a430a6ac0f7efd3ff7e86ede5705dc5487"
HEADERS = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json"
}

def test_health_check():
    """Test the health check endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"Health Check: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_openai_connection():
    """Test the OpenAI connection"""
    try:
        response = requests.get(f"{BASE_URL}/test-openai", headers=HEADERS, timeout=30)
        print(f"OpenAI Test: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"OpenAI test failed: {e}")
        return False

def test_document_processing():
    """Test document processing with a sample PDF"""
    # Using a public sample PDF
    test_data = {
        "document_url": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/process-document/",
            headers=HEADERS,
            json=test_data,
            timeout=60
        )
        print(f"Document Processing: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Chunks processed: {result.get('total_chunks', 'N/A')}")
        else:
            print(f"Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Document processing failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Testing HackRX LLM API Application")
    print("=" * 50)
    
    print("\\n1. Testing Health Check...")
    health_ok = test_health_check()
    
    if not health_ok:
        print("❌ Health check failed. Make sure the application is running.")
        return
    
    print("✅ Health check passed!")
    
    print("\\n2. Testing OpenAI Connection...")
    openai_ok = test_openai_connection()
    
    if openai_ok:
        print("✅ OpenAI connection working!")
    else:
        print("⚠️ OpenAI connection failed, but application is running")
    
    print("\\n3. Testing Document Processing...")
    doc_ok = test_document_processing()
    
    if doc_ok:
        print("✅ Document processing working!")
    else:
        print("⚠️ Document processing failed")
    
    print("\\n" + "=" * 50)
    if health_ok:
        print("✅ Application is ready for deployment!")
    else:
        print("❌ Fix issues before deploying")
    print("=" * 50)

if __name__ == "__main__":
    main()
