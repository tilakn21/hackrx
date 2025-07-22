# Render Deployment Guide for HackRX LLM API

## Prerequisites
1. GitHub account
2. Render account (free)
3. Your code should be pushed to a GitHub repository

## Step-by-Step Deployment Instructions

### 1. Prepare Your Repository
Make sure all the following files are in your repository:
- `main.py` - Your FastAPI application
- `config.py` - Configuration file
- `requirements.txt` - Python dependencies
- `render.yaml` - Render configuration
- `Procfile` - Process file for Render
- `runtime.txt` - Python version specification

### 2. Create Render Service

1. **Go to Render Dashboard**
   - Visit https://dashboard.render.com/
   - Sign in with your GitHub account

2. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repository containing your code

3. **Configure Service Settings**
   - **Name**: `hackrx-llm-api` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `python main.py`
   - **Plan**: Select `Free` tier

4. **Environment Variables**
   Add these environment variables in Render dashboard:
   ```
   PYTHONUNBUFFERED=1
   PORT=10000
   PYTHONPATH=.
   ```

5. **Advanced Settings**
   - **Health Check Path**: `/health`
   - **Auto-Deploy**: Enable if you want automatic deployments on git push

### 3. Deploy
1. Click "Create Web Service"
2. Render will automatically start building and deploying your app
3. The build process may take 5-10 minutes on the first deployment
4. Monitor the logs for any errors

### 4. Verify Deployment
Once deployed, you can test your API:

1. **Health Check**
   ```
   GET https://your-app-name.onrender.com/health
   ```

2. **Test Document Processing**
   ```
   POST https://your-app-name.onrender.com/process-document/
   Content-Type: application/json
   Authorization: Bearer 4e604cba5381f75493ad0742118a94a430a6ac0f7efd3ff7e86ede5705dc5487
   
   {
     "document_url": "https://example.com/sample.pdf"
   }
   ```

3. **Test Query**
   ```
   POST https://your-app-name.onrender.com/query/
   Content-Type: application/json
   Authorization: Bearer 4e604cba5381f75493ad0742118a94a430a6ac0f7efd3ff7e86ede5705dc5487
   
   {
     "query": "What is this document about?",
     "document_url": "https://example.com/sample.pdf"
   }
   ```

## Important Notes for Free Tier

### Limitations
- **Memory**: 512 MB RAM limit
- **Build Time**: 15 minutes maximum
- **Sleep Mode**: Service sleeps after 15 minutes of inactivity
- **Cold Start**: May take 10-30 seconds to wake up from sleep

### Optimizations Applied
1. **Single Worker**: Configured to use 1 worker to avoid memory issues
2. **CPU-Only PyTorch**: Using `torch==2.1.1+cpu` for smaller memory footprint
3. **Efficient Model**: Using `all-MiniLM-L6-v2` (lightweight sentence transformer)
4. **Cleanup**: Temporary files are properly managed

### Troubleshooting

**Build Failures:**
- Check that all dependencies in requirements.txt are compatible
- Monitor build logs for specific error messages
- Ensure Python version is 3.9.x

**Memory Issues:**
- Service may restart if memory usage exceeds 512MB
- Consider reducing batch sizes or model complexity
- Monitor service metrics in Render dashboard

**Cold Start Issues:**
- First request after sleep may timeout
- Consider using a monitoring service to keep it warm
- Health check endpoint helps with service monitoring

## Alternative: Using render.yaml
If you prefer infrastructure-as-code, you can use the `render.yaml` file in your repository root. Render will automatically detect and use this configuration.

## Support
- Render Documentation: https://render.com/docs
- FastAPI Documentation: https://fastapi.tiangolo.com/
- Your application logs are available in the Render dashboard

## Security Note
Remember to:
- Keep your API keys secure
- Use environment variables for sensitive data
- Consider implementing rate limiting for production use
