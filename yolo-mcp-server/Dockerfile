FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy both the MCP server and the main yolo-code-assistant project
COPY yolo-mcp-server /app/yolo-mcp-server
COPY yolo-code-assistant /app/yolo-code-assistant

# Set working directory to MCP server
WORKDIR /app/yolo-mcp-server

# Copy requirements and install dependencies
COPY yolo-mcp-server/pyproject.toml .
RUN pip install --no-cache-dir uv && \
    uv pip install --system -r pyproject.toml

# Install additional dependencies that might be missing
RUN pip install --no-cache-dir \
    fastmcp \
    pymongo \
    sentence-transformers \
    openai \
    python-dotenv \
    gitpython \
    tqdm \
    tree-sitter \
    tree-sitter-python

# Set Python path to include parent directory
ENV PYTHONPATH=/app:$PYTHONPATH

# Run the MCP server
CMD ["python", "mcp_server.py"]
