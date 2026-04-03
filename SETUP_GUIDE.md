# CropAI: Installation and Setup Guide

Welcome to **CropAI**, an intelligent, containerized agricultural platform powered by FastAPI, React, and Machine Learning. 

This document serves as the complete technical manual for developers, maintainers, and users looking to set up, configure, and run the CropAI project.

---

## Table of Contents
1. [Prerequisites](#1-prerequisites)
2. [Project Structure](#2-project-structure)
3. [Method 1: Running via Docker (Recommended, Easiest)](#3-method-1-running-via-docker)
4. [Method 2: Manual Local Development Setup](#4-method-2-manual-local-development-setup)
5. [Using the Application](#5-using-the-application)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Prerequisites

Before beginning, ensure your system has the following installed:
- **Git** (for version control and cloning)
- **Node.js**: `v18.0.0` or higher (required for React + Vite development)
- **Python**: `v3.9` or higher (tested meticulously on Python 3.10)
- **Docker & Docker Compose** (Optional, but highly recommended for 1-click deployment)

---

## 2. Project Structure

When you open the CropAI root directory, it is structured as an integrated monorepo:

```text
CropAI/
├── api.py                  # Main entry point for FastAPI backend
├── database.py             # SQLite database logic and query functions
├── config.py               # Application configuration and variables
├── disease_model.py        # Logic for ML image classification
├── yield_model.py          # Logic for Ensemble ML regression
├── requirements.txt        # Python pip dependencies
├── docker-compose.yml      # Docker orchestration file
├── nginx.conf              # Reverse proxy configurations
├── frontend/               # <--- NEW REACT DASHBOARD DIRECTORY
│   ├── package.json        # Node dependencies
│   ├── vite.config.ts      # Vite bundler configuration
│   ├── tailwind.css        # Tailwind V4 utilities
│   └── src/
│       ├── store/          # Zustand global state management
│       ├── services/       # Axios API wrapper functions
│       └── pages/          # React specific routes/pages
└── models/                 # Directory where .h5 and .pkl weights live
```

---

## 3. Method 1: Running via Docker

If you want the production experience without installing localized Python or Node dependency chains, use Docker. 

The `docker-compose.yml` file is configured to execute a multi-container build:
1. **api:** Builds the FastAPI container.
2. **frontend:** Builds a statically compiled NGINX server serving the React Vite build.
3. **nginx:** A reverse proxy binding Port 80 to route `/api` traffic to FastAPI, and all other traffic to the React UI.

### Step-by-step
1. Open a terminal at the project root (`CropAI/`).
2. Run the build and boot command:
   ```bash
   docker-compose up --build -d
   ```
3. Open your browser and navigate to:
   - Dashboard: `http://localhost` or `http://localhost:80`
   - API Docs: `http://localhost/api/docs`
4. To stop the containers, run:
   ```bash
   docker-compose down
   ```

---

## 4. Method 2: Manual Local Development Setup

If you wish to edit the code, develop new React components, or tweak the Python AI models, you must run the systems manually in development mode.

### Part A: Python Backend (FastAPI) Setup
1. **Navigate to the core directory**:
   ```bash
   cd CropAI
   ```
2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   ```
3. **Activate the Virtual Environment**:
   - On **Windows**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - On **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```
4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Start the FastAPI Server**:
   ```bash
   python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload
   ```
   *(The `--reload` flag means the server will reboot automatically if you edit Python files.)*

### Part B: React Frontend Setup
1. **Open a NEW terminal window/tab**.
2. **Navigate to the frontend directory**:
   ```bash
   cd CropAI/frontend
   ```
3. **Install Node Packages**:
   ```bash
   npm install
   ```
4. **Start the Vite Development Server**:
   ```bash
   npm run dev
   ```
5. Vite will launch locally, usually at `http://localhost:5173`. 
   *(Note: The `vite.config` and Axios `api.ts` file automatically point their API calls to your local `http://localhost:8000` Python server you started in Part A.)*

---

## 5. Using the Application

Once both environments are up and running, you can access the platform via your browser.

- **Dashboard Home**: This displays overall analytics. It will be empty initially until your first prediction is made.
- **Disease Intelligence (Computer Vision)**:
  1. Click to the Disease Intelligence tab on the sidebar.
  2. Drag and drop a clear picture of a crop leaf into the upload bounded square, or click "Browse Files".
  3. The system will process it, and present a card with the prediction, confidence level, severity index, and a recommended chemical or organic treatment.
- **Yield Intelligence (Agro-Metrics)**:
  1. Click the Yield Intelligence tab.
  2. Fill out the agronomic form parameters (crop type, area size, fertilizers, etc.).
  3. Select **"Generate Forecast"**. The application will compute a highly localized estimation of yields (kg/ha) and output corresponding smart-guidelines.
- **Analysis History**:
  1. Navigate to the History/Activity Log tab.
  2. Here you can search, view, and analyze all predictions that have been committed to the local SQLite database.

---

## 6. Detailed Troubleshooting & FAQ

**Q1: The Frontend shows a "Backend Unreachable" error.**
- **Check CORS**: The FastAPI app in `api.py` has CORS middleware. Ensure `allow_origins=["*"]` is present if you are debugging across different network interfaces.
- **Vite Proxy**: If you aren't using the default `localhost:8000`, update the `VITE_API_URL` in your environment or directly in `frontend/src/services/api.ts`.
- **Port Conflict**: Run `netstat -ano | findstr :8000` on Windows or `lsof -i :8000` on Linux/Mac to see if another process is blocking the API.

**Q2: Predictions are slow or timing out.**
- **Hardware**: Deep Learning models (EfficientNet) are computationally expensive. On a CPU-only machine, expect ~200-500ms per image. If using a GPU, ensure `torch.cuda.is_available()` returns `True`.
- **Workers**: In `config.py`, check `NUM_WORKERS`. For most development machines, `0` or `2` is optimal.

**Q3: How do I retrain the model with my own data?**
1. Empty the `data/plantvillage` directory.
2. Place your images in `data/plantvillage/<ClassName>/` folders.
3. Run `python disease_model.py`. The script will detect the new data, generate a new `class_mapping.json`, and start the training loop automatically.

---

## 7. Operational Checklist for Production
- [ ] **Environment Variables**: Create a `.env` file in the root with `API_HOST`, `API_PORT`, and `LOG_LEVEL`.
- [ ] **Docker Health**: Run `docker ps` to ensure both `api` and `frontend` containers are "Up".
- [ ] **Storage**: Ensure the `data/` directory has write permissions for the SQLite database.
- [ ] **Logs**: Monitor `logs/api.log` for any runtime exceptions or high-latency warnings.

---

**Happy Farming!** 🌾
*For further contributions or bug reporting, please refer to the project repository or contact the lead maintainer.*
