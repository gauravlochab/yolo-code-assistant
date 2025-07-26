# 🚀 YOLO Code Assistant - Complete RAG System

A comprehensive RAG-based code assistant system for the Ultralytics YOLO codebase, featuring both a standalone application and MCP server integration for AI assistants.

## 📁 Project Structure

```
YOLO-RAG/
├── yolo-code-assistant/     # Main RAG application
│   ├── src/                 # Core application code
│   ├── tests/              # Test suite
│   ├── notebooks/          # Jupyter exploration notebooks
│   └── main.py             # Application entry point
├── yolo-mcp-server/        # MCP server for AI assistants
│   ├── mcp_server.py       # MCP protocol implementation
│   └── README.md           # MCP-specific documentation
└── README.md               # This file
```

## 🎯 Components Overview

### 1. YOLO Code Assistant (Main Application)
A sophisticated RAG system that provides intelligent code assistance for the Ultralytics YOLO codebase.

**Key Features:**
- **Tree-sitter** based Python code parsing
- **Jina embeddings** for semantic code understanding  
- **MongoDB Atlas** vector storage and search
- **OpenRouter API** integration with free models
- **Gradio** web interface for interactive queries
- **Comprehensive testing** suite with YOLO-specific mocks

### 2. YOLO MCP Server (AI Assistant Integration)
A Model Context Protocol server that enables integration with AI assistants like Claude Desktop.

**Key Features:**
- **MCP Protocol** compliant server implementation
- **Specialized YOLO tools** for code generation and assistance
- **Claude Desktop** integration support
- **Docker** containerization support

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- MongoDB Atlas account (free M0 cluster)
- OpenRouter API key (free tier available)

### 1. Main Application Setup

```bash
cd yolo-code-assistant

# Install dependencies
make install

# Setup environment
cp .env.sample .env
# Edit .env with your MongoDB and OpenRouter credentials

# Index the YOLO codebase
make index

# Launch web interface
make serve
```

### 2. MCP Server Setup

```bash
cd yolo-mcp-server

# Install dependencies
uv sync

# Setup environment
cp .env.example .env
# Edit .env with your credentials

# Test server
uv run python mcp_server.py
```

## 🏗️ Architecture

### RAG Pipeline
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Gradio    │────▶│   Retrieval  │────▶│  MongoDB    │
│     UI      │     │    Engine    │     │   Atlas     │
└─────────────┘     └──────────────┘     └─────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │  OpenRouter  │
                    │     LLM      │
                    └──────────────┘
```

### MCP Integration
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Claude    │────▶│  MCP Server  │────▶│    RAG      │
│  Desktop    │     │   Protocol   │     │   System    │
└─────────────┘     └──────────────┘     └─────────────┘
```

## 🔧 Development

### Running Tests
```bash
cd yolo-code-assistant
make test

# Or run specific test components
cd tests && uv run python simple_test.py
```

### Development Mode
```bash
cd yolo-code-assistant
make dev  # Runs check + serve
```

### Jupyter Notebooks
```bash
cd yolo-code-assistant
make notebook
```

## 📊 Performance Metrics

- **Indexing Speed**: ~500 files/minute
- **Search Latency**: <200ms average
- **Embedding Generation**: ~100 chunks/second
- **Response Time**: 2-5 seconds (depends on OpenRouter)
- **Database**: 949+ indexed code chunks



**Built with ❤️ for intelligent code assistance**

