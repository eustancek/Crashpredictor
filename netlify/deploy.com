#!/bin/bash
# Build frontend
npm run build

# Deploy to Netlify
netlify deploy --prod

# Deploy backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port $PORT