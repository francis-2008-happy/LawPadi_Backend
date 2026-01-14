# LawPadi Backend - Railway Deployment Guide

## Prerequisites

1. **GitHub Account** - Your code needs to be on GitHub
2. **Railway Account** - Sign up at [railway.app](https://railway.app)
3. **Groq API Key** - For LLM generation (store in Railway environment variables)

---

## Step 1: Prepare Your Repository

### 1.1 Ensure vectorstore is NOT committed (large files)

The `.gitignore` already excludes `vectorstore/`, which is good since Railway has file size limits.

### 1.2 Commit deployment files

```bash
git add Procfile railway.toml runtime.txt
git add requirements.txt .gitignore
git commit -m "Add Railway deployment configuration"
git push origin main
```

---

## Step 2: Deploy to Railway

### 2.1 Create New Project

1. Go to [railway.app](https://railway.app)
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Authorize Railway to access your GitHub
5. Select your `LawPadi_Backend` repository

### 2.2 Configure Environment Variables

After deployment starts, click on your service → **Variables** tab:

**Required Environment Variables:**

```bash
# Groq API Key (required for answer generation)
GROQ_API_KEY=your_groq_api_key_here

# Python environment
PYTHONUNBUFFERED=1

# Optional: If you use other APIs
# OPENAI_API_KEY=your_key_here
```

### 2.3 Configure Build Settings (if needed)

Railway auto-detects Python projects. The settings are in `railway.toml`:
- **Build Command:** Auto-detected (`pip install -r requirements.txt`)
- **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Python Version:** 3.11 (from `runtime.txt`)

---

## Step 3: Build the Index on Railway

**Important:** The vectorstore files are large and excluded from git. You need to build the index on Railway.

### Option A: One-time Build Script (Recommended)

Add a build script that runs after dependencies are installed:

1. Click your service → **Settings** → **Custom Build Command**
2. Add:
   ```bash
   pip install -r requirements.txt && python -m app.rag.build_index
   ```

**Note:** This will run during deployment and may take 5-10 minutes the first time.

### Option B: Manual Build via Railway Shell

1. After deployment, click on your service
2. Go to **Settings** → **Enable Railway Shell**
3. Open the shell and run:
   ```bash
   python -m app.rag.build_index
   ```

### Option C: Use Railway Volumes (For persistence)

If you want the index to persist across deployments:

1. Go to your service → **Settings** → **Volumes**
2. Add a volume mounted at `/app/vectorstore`
3. Build the index once (via Option B)
4. Index will persist across redeployments

**Railway Volume Setup:**
```
Mount Path: /app/vectorstore
Size: 2GB (minimum for your 85MB vectorstore)
```

---

## Step 4: Upload Legal Documents

Since your `data/` folder contains large PDFs, you have options:

### Option A: Commit smaller datasets
```bash
# If your data folder is < 100MB total
git add data/
git commit -m "Add legal documents"
git push
```

### Option B: Use Railway Volumes for data
1. Create another volume at `/app/data`
2. Upload files via Railway shell or API

### Option C: Use Cloud Storage
Store PDFs in AWS S3, Google Cloud Storage, or similar, and download during index build.

---

## Step 5: Verify Deployment

### 5.1 Check Build Logs

1. Click on your service
2. View **Deployments** → Latest deployment → **View Logs**
3. Ensure no errors during:
   - Dependency installation
   - Index building
   - Server startup

### 5.2 Test API Endpoints

Railway will provide a URL like: `https://your-app.up.railway.app`

Test endpoints:
```bash
# Health check
curl https://your-app.up.railway.app/docs

# Query endpoint
curl -X POST https://your-app.up.railway.app/query/ \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the penalty for theft in Nigeria?"}'
```

---

## Step 6: Domain Configuration (Optional)

### Custom Domain

1. Go to your service → **Settings** → **Domains**
2. Click **Generate Domain** (get a railway.app subdomain)
3. Or add your **Custom Domain** and configure DNS

---

## Important Notes

### File Persistence

⚠️ **Railway ephemeral filesystem**: Files written to disk are lost on redeployment unless using Volumes.

**Solution:**
- Use Railway Volumes for `vectorstore/` (recommended)
- Or rebuild index on each deployment (slower)

### Memory & Resources

Your app requires:
- **Memory:** ~1-2GB (for sentence-transformers model + FAISS index)
- **Disk:** ~500MB (dependencies) + 85MB (vectorstore) + documents

Railway's free tier provides:
- 512MB RAM (may need to upgrade to Hobby plan)
- 1GB disk space

**Recommendation:** Start with **Hobby plan** ($5/month) for 8GB RAM.

### Cost Optimization

1. **Disable auto-deployments** if you don't need them
2. **Use sleep mode** for development (Railway can pause after inactivity)
3. **Monitor usage** in Railway dashboard

---

## Troubleshooting

### Build Fails: "No module named 'faiss'"

**Solution:** Railway is installing dependencies correctly. If using custom build command, ensure:
```bash
pip install -r requirements.txt && python -m app.rag.build_index
```

### Out of Memory during build_index

**Solution:** 
- Upgrade to Hobby plan (8GB RAM)
- Or reduce batch size in `build_index.py`:
  ```python
  # In embeddings, process in smaller batches
  embedder.transform(chunks, batch_size=32)  # Default is 32
  ```

### Vectorstore not found at runtime

**Solution:**
- Ensure index is built (check deployment logs)
- Or set up Railway Volume for persistence
- Check `vectorstore/` exists: `ls -la vectorstore/` in Railway shell

### GROQ_API_KEY not working

**Solution:**
- Verify environment variable is set in Railway dashboard
- Restart deployment after adding variables
- Check logs for API errors

---

## Quick Start Commands

```bash
# 1. Commit deployment files
git add Procfile railway.toml runtime.txt
git commit -m "Add Railway deployment config"
git push origin main

# 2. Deploy on Railway
# - Go to railway.app
# - New Project → Deploy from GitHub
# - Select LawPadi_Backend

# 3. Set environment variables in Railway dashboard:
#    GROQ_API_KEY=your_key

# 4. Build index (via Railway shell):
python -m app.rag.build_index

# 5. Test your API:
curl https://your-app.up.railway.app/docs
```

---

## Files Created for Railway

- ✅ `Procfile` - Defines start command
- ✅ `railway.toml` - Railway-specific configuration
- ✅ `runtime.txt` - Python version specification
- ✅ `requirements.txt` - Already exists with all dependencies

---

## Next Steps After Deployment

1. **Monitor logs** for any runtime errors
2. **Test all endpoints** (query, ingest)
3. **Set up health checks** (Railway auto-detects FastAPI)
4. **Configure custom domain** if needed
5. **Set up CI/CD** for automatic deployments on git push

---

## Support & Resources

- **Railway Docs:** https://docs.railway.app
- **Railway Discord:** https://discord.gg/railway
- **Python on Railway:** https://docs.railway.app/guides/python

---

Need help? Check the deployment logs in Railway dashboard or run diagnostics:

```bash
# In Railway shell
python -c "import faiss; import groq; import sentence_transformers; print('✅ All imports working')"
ls -lh vectorstore/
```
