# Power System Analysis Chatbot

An intelligent multi-agent chatbot system for power system analysis tasks, leveraging Large Language Models (LLMs) and MATLAB integration for complex electrical engineering computations.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2024a+-orange.svg)](https://www.mathworks.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.51.0-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Table of Contents

- [Features](#-features)
- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Technology Stack](#️-technology-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- **Knowledge Graph Pipeline** - Automated PDF ingestion (text & images) into a Neo4j graph database for advanced context retrieval.
- **MATLAB Code Execution** - General-purpose MATLAB code generation and execution for power system analysis (load flow, faults, losses), control systems, and mathematical computations.
- **General Web Search** - Answering broader power system questions via DuckDuckGo and iterative LLM synthesis.
- **Multimodal Input Support** - Text and image inputs for enhanced analysis.
- **Intelligent Query Routing** - Automatic classification and routing to appropriate agents or graph search.

## 🎯 Project Overview

This project implements an AI-powered assistant capable of handling various power system analysis tasks and creating automated knowledge graphs from technical literature. The system uses a sophisticated multi-agent architecture with intelligent query routing and supports multimodal inputs.

The chatbot leverages:
- **Ollama** for local, private, and fast LLM inference (replacing external APIs)
- **MATLAB Engine API** for numerical computations
- **Neo4j** for vector and relationship-based Knowledge Graph storage
- **Streamlit** for the web interface
- **Multi-agent pattern** for specialized task handling

## 🏗️ Architecture

### System Architecture Diagram

```text
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend Layer                           │
│                    (Streamlit Web UI)                           │
│              - Chat Interface & Image Upload                    │
│              - Knowledge Graph Ingestion Sidebar                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestrator Layer                           │
│                   (orchestrator.py)                             │
│                                                                 │
│  ┌──────────────────────────────────────────────────┐           │
│  │ LLM Query Classifier (Ollama)                    │           │
│  │ - Analyzes user intent & Routes query            │           │
│  │ - Handles context & Multimodal inputs            │           │
│  └──────────────────────────────────────────────────┘           │
│                         │                                       │
│            ┌────────────┴────────────┐                          │
│            ▼                         ▼                          │
│   ┌─────────────────┐      ┌──────────────────┐                 │
│   │ MATLAB Executor │      │ Web Search Agent │                 │
│   │ Agent           │      │ (Iterative)      │                 │
│   └────────┬────────┘      └────────┬─────────┘                 │
└────────────┼────────────────────────┼───────────────────────────┘
             │                        │
             ▼                        ▼
┌──────────────────────┐   ┌──────────────────────┐
│  MATLAB Computation  │   │  Search & Knowledge  │
│  Layer               │   │  Layer               │
│                      │   │                      │
│ - MATLAB Engine API  │   │ - DuckDuckGo API     │
│ - Script generation  │   │ - Neo4j Graph DB     │
│ - Code execution     │   │ - kg_pipeline.py     │
└──────────────────────┘   └──────────────────────┘
```

### Key Components

### Key Components

#### 1. **Orchestrator** (`orchestrator.py`)
The central routing system that:
- Uses local Ollama LLMs with tool calling to classify user queries.
- Routes queries to either the `matlab_executor` or `web_search` agents.
- Contextualizes historical conversation data into standalone prompts.
- Supports multimodal inputs (text + base64 encoded images).

#### 2. **MATLAB Executor Agent** (`agents/matlab_executor_agent.py`)
Intelligent MATLAB code generation and execution system replacing legacy hardcoded agents:
- **LLM-Powered Code Generation** - Generates dynamic MATLAB scripts for any technical query (e.g., Ybus calculation, power flow, control systems).
- **Dual Execution Modes**:
  - **Calculation Mode**: Executes MATLAB code and returns text output.
  - **Plotting Mode**: Extracts plot data from MATLAB workspace and creates matplotlib visualizations.
- **Iterative Refinement** - Automatically self-corrects code errors up to 5 times.
- **File I/O** - Seamlessly handles `.csv` file reads for dataset-driven MATLAB problems.

#### 3. **Knowledge Graph Pipeline** (`kg_pipeline.py`)
Automated ingestion pipeline for building a context-aware graph:
- **PDF Processing** - Downloads and chunks PDF files into logical text blocks.
- **Multimodal Extraction** - Uses Ollama Vision (`llava`) to extract insights directly from PDF page images.
- **Entity & Relationship Mining** - Extracts structured nodes (entities) and edges (relationships) using LLMs.
- **Neo4j Storage** - Persists the network into a Neo4j graph database for advanced querying and RAG (Retrieval-Augmented Generation).

> **Note on Customization:** The extraction `PROMPT` in `kg_pipeline.py` is currently configured for general structural analysis. You should modify this prompt based on the specific type of Knowledge Graph you want to build to ensure the extracted entities and relationships are relevant to your domain.

#### 4. **Web Search Agent** (`agents/websearch_agent.py`)
- Uses the DuckDuckGo API for live web searches.
- Connects directly to the Neo4j Knowledge Graph to answer domain-specific questions.
- Uses an iterative loop to assess if enough information has been gathered before synthesizing a final answer.

> **Note on Customization:** The query routing logic in `agents/websearch_agent.py` (specifically the `route_query_to_source` function) currently has a hardcoded prompt for a specific domain (e.g., routing queries related to "NIT Andhra Pradesh"). You must change this prompt here to match the domain of the Knowledge Graph you have ingested.

#### 5. **Frontend** (`app.py`)
Streamlit-based web interface with:
- Chat-style conversation UI and session history.
- Image upload support.
- **Sidebar Integration** - Direct UI for pasting PDF URLs to trigger the `kg_pipeline.py` ingestion process.

---

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.10+** - Primary programming language
- **Ollama** - Local LLM inference engine (`llama3`, `llava`)
- **Neo4j** - Graph Database for Knowledge Graph storage
- **MATLAB Engine API** - Numerical computations

### Agent Framework
- **Tool Calling** - LLM-based function calling for agent coordination
- **Multi-Agent Pattern** - Hierarchical agent architecture

### Web & Search
- **Streamlit** - Web interface framework
- **DuckDuckGo (ddgs)** - Web search API
- **Pillow** - Image processing

### Development Tools
- **python-dotenv** - Environment variable management
- **Git** - Version control

---

## 📦 Installation

### Prerequisites
1. **Python 3.10 or higher**
2. **MATLAB R2024a or higher** (with a valid license)
3. **Docker Desktop** (for running Neo4j)
4. **Ollama** (for running local LLMs)

### Setup Instructions

#### 1. Clone the repository
```bash
git clone https://github.com/yourusername/Major_Project.git
cd Major_Project/chatbot
```

#### 2. Virtual Environment and Requirements Download
It is highly recommended to isolate the project dependencies using a virtual environment.
```bash
# Create the virtual environment
python -m venv myenv

# Activate the virtual environment
# On Windows:
myenv\Scripts\activate
# On Linux/Mac:
source myenv/bin/activate

# Install all required Python dependencies
pip install -r requirements.txt
```

#### 3. MATLAB Setup
To allow Python to communicate with MATLAB, you must install the MATLAB Engine API for Python.
1. Locate your MATLAB installation root folder (e.g., `C:\Program Files\MATLAB\R2024a`).
2. Navigate to the Python engine setup directory in your terminal (make sure your virtual environment is active):
```bash
cd "C:\Program Files\MATLAB\R2024a\extern\engines\python"
```
3. Install the engine:
```bash
python setup.py install
```
*(Note: If you face permission errors on Windows, run your command prompt as Administrator.)*

#### 4. Neo4j Server using Docker
We use Neo4j to store the Knowledge Graph generated from PDFs.
1. Ensure Docker is running on your system.
2. Pull the Neo4j Docker image and start the container with the following command:
```bash
docker run \
    --name neo4j-chatbot \
    -p 7474:7474 -p 7687:7687 \
    -d \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:latest
```
3. You can verify it is running by navigating to `http://localhost:7474` in your web browser.

#### 5. How to use Ollama
Ollama is used to run the LLMs locally, replacing external APIs like Groq.
1. Download and install [Ollama](https://ollama.com/) for your operating system.
2. Open a terminal and pull the required models by running:
```bash
ollama run llama3
ollama run llava   # Only needed if you plan to use image analysis in the KG pipeline
```
3. Ensure the Ollama background service is running (by default, it runs at `http://localhost:11434`).

#### 6. Configure Environment Variables
Create a `.env` file in the project `chatbot/` root folder and configure your connection settings:
```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
OLLAMA_VISION_MODEL=llava

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

---

## 🚀 Usage

### Running the Web Interface
```bash
streamlit run app.py
```
The application will open in your browser at `http://localhost:8501`

### Running the CLI Version
```bash
python orchestrator.py
```

### Example Queries

**Power Flow Analysis**:
```
Calculate bus voltages for a 3-bus system with the following data:
- Branch 1: From bus 1 to bus 2, R=0.03, X=0.08, shunt=0.04
- Branch 2: From bus 1 to bus 3, R=0.02, X=0.05, shunt=0.02
- Bus 1: Slack bus, V=1.0∠0°
- Bus 2: PQ bus, P=-1.5, Q=-0.5
- Bus 3: PV bus, P=-2.0, V=1.02
```

**Fault Analysis**:
```
Find post-fault voltages for a three-phase fault at bus 2.
Pre-fault voltages: [1.0+0j, 0.95-0.1j, 0.98-0.05j]
Ybus: [[10-20j, -5+10j, -5+10j], ...]
```

**Loss Calculation**:
```
Calculate system losses after adding a load of 0.5+0.2j pu at bus 3.
Current voltages: [1.0, 0.95∠-5°, 0.98∠-3°]
```

**MATLAB Code Execution**:
```
Plot the step response of the transfer function H(s) = 5 / (s^2 + 3s + 2)
```

```
Create a 3x3 matrix with values [[1,2,3],[4,5,6],[7,8,9]] and calculate its determinant and eigenvalues.
```

**Web Search**:
```
What is the difference between Newton-Raphson and Gauss-Seidel methods?
```

### Using Image Input
1. Click the 📎 "Attach image" button
2. Upload a circuit diagram or system schematic
3. Ask a question about the image
4. The system will analyze the image and provide context-aware responses

---

## 📂 Project Structure

```
Major_Project/
├── chatbot/
│   ├── agents/                          # Agent modules
│   │   ├── __init__.py
│   │   ├── matlab_executor_agent.py     # MATLAB code generation & execution
│   │   └── websearch_agent.py           # Web search agent (Web + Neo4j)
│   │
│   ├── matlab_scripts/                  # Historical MATLAB computation scripts
│   │   └── ...
│   │
│   ├── orchestrator.py                  # Main orchestrator with query routing
│   ├── app.py                           # Streamlit web interface
│   ├── kg_pipeline.py                   # Knowledge Graph PDF ingestion pipeline
│   │
│   ├── requirements.txt                 # Python dependencies
│   ├── .env                             # Environment variables (not in git)
│   └── README.md                        # This file
│
├── project_report.pdf                   # Project documentation
└── ...
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 🙏 Acknowledgments

- Ollama and the open-source AI community for providing robust, local LLM infrastructure.
- Neo4j for powerful graph database capabilities.
- All contributors and users of this project

---

Made with ❤️ and ⚡ by the Power Systems Team
