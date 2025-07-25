#!/bin/bash

# Set the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the MCP server directory
cd "$SCRIPT_DIR"

# Load .env file if it exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Override with any environment variables passed in
export MONGODB_URI="${MONGODB_URI}"
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY}"

# Run the server with uv
exec uv run python mcp_server.py
