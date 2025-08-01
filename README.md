# Turbidity Classifier App

## Overview

This repository contains:
- **Backend**: Flask service that loads a trained ResNet-34 turbidity classifier and exposes `/predict`.
- **Mobile**: React Native (Expo) app to take/upload water photos, send to backend, and display predictions.


## Backend Setup

```bash
cd backend
python -m pip install --upgrade pip
pip install -r requirements.txt
# Place best_turbidity_model.pth in this folder
python app.py
```

The backend will listen on port 5000:
`http://<your-machine-ip>:5000/predict`

## Mobile Setup (Expo)

```bash
cd mobile
npm install
expo start
```

Edit `App.js` and replace `BACKEND_URL` with your backend IP (e.g., `http://192.168.1.100:5000/predict`).

## GitHub

To push:

```bash
git init
git add .
git commit -m "Initial commit"
# create repo on GitHub manually, then:
git remote add origin <your-github-url>
git push -u origin main
```
