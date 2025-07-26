# YOLO Code Assistant MCP Server

An MCP (Model Context Protocol) server that provides YOLO code generation tools for AI assistants like Claude Desktop.

## Features

This MCP server provides four specialized tools for YOLO development:

1. **`generate_yolo_code`** - Generate complete Python code for YOLO tasks
2. **`ask_yolo_question`** - Answer technical questions about YOLO
3. **`improve_yolo_code`** - Optimize and improve existing YOLO code
4. **`explain_yolo_concept`** - Explain YOLO concepts with code examples

## Prerequisites

- Python 3.11 or higher
- UV package manager (`pip install uv`)
- MongoDB Atlas account (for vector storage)
- OpenRouter API key (for LLM generation)
- The YOLO codebase should already be indexed using the main `yolo-code-assistant` project

## Setup

### 1. Clone and Navigate to Directory

```bash
cd <repo-path>
```

### 2. Create Environment File

Copy the example environment file and add your credentials:

```bash
cp .env.example .env
```

Edit `.env` and add your actual credentials:
```
MONGODB_URI=mongodb+srv://your-actual-connection-string
OPENROUTER_API_KEY=sk-or-your-actual-api-key
```

### 3. Install Dependencies

```bash
uv sync
```

### 4. Test the Server

Run the server manually to verify it works:

```bash
uv run python mcp_server.py
```

You should see:
```
Starting YOLO Code Assistant MCP Server...
Available tools:
  - generate_yolo_code: Generate complete YOLO code for a task
  - ask_yolo_question: Answer technical YOLO questions
  - improve_yolo_code: Improve existing YOLO code
  - explain_yolo_concept: Explain YOLO concepts with examples
```

Press `Ctrl+C` to stop the test.

## Claude Desktop Integration

### 1. Find Claude Desktop Config

On macOS:
```bash
~/Library/Application Support/Claude/claude_desktop_config.json
```

### 2. Edit Configuration

Add the YOLO MCP server to your config:

```json
{
  "mcpServers": {
    "yolo-assistant": {
      "command": "uv",
      "args": ["run", "python", "/Users/gauravlochab/YOLO-RAG/yolo-mcp-server/mcp_server.py"]
    }
  }
}
```

### 3. Restart Claude Desktop

Quit and restart Claude Desktop for the changes to take effect.

## Usage Examples

Once integrated with Claude Desktop, you can use the tools like this:

### Generate YOLO Code
```
"Generate code to detect people in a webcam stream using YOLO"
```

### Ask Technical Questions
```
"How do I train YOLO on a custom dataset with 10 classes?"
```

### Improve Existing Code
```
"Improve this YOLO inference code: [paste your code]"
```

### Explain Concepts
```
"Explain how Non-Maximum Suppression (NMS) works in YOLO"
```

## Troubleshooting

### Server Won't Start

1. Check environment variables are set correctly in `.env`
2. Verify MongoDB connection string is valid
3. Ensure OpenRouter API key is active

### Claude Desktop Can't Find Server

1. Verify the full path in `claude_desktop_config.json` is correct
2. Ensure the server has execute permissions
3. Check Claude Desktop logs for errors

### Import Errors

The server expects your directory structure as:
```
yolo-code-assistant/
├── yolo-code-assistant/
│   └── src/
└── yolo-mcp-server/
    └── mcp_server.py
```

## Development

To modify the server:

1. Edit `mcp_server.py`
2. Test changes with `uv run python mcp_server.py`
3. Restart Claude Desktop to load changes

## Sharing with Others

**IMPORTANT:** Never share your MongoDB URI or API keys directly!

To share this project with others:

1. **Direct them to use the template:**
   ```bash
   cp .env.template .env
   # Then edit .env with their own credentials
   ```

2. **Tell them what they need:**
   - MongoDB Atlas account (free tier works)
   - OpenRouter API key
   - The database is already indexed with YOLO code

3. **Share these details only:**
   - Repository URL
   - Database name: `yolo_assistant` (if needed)
   - Collection name: `code_chunks` (if needed)

4. **For detailed security practices:**
   See `SHARING_MONGODB_ACCESS.md` for comprehensive security guidelines

### Quick Setup Message Template
```
To use the YOLO Code Assistant MCP Server:

1. Clone: https://github.com/yourusername/yolo-code-assistant
2. Get MongoDB Atlas account: https://www.mongodb.com/cloud/atlas
3. Get OpenRouter API key: https://openrouter.ai/
4. Setup: cp .env.template .env (then add your credentials)
5. Run: uv sync && uv run python mcp_server.py

The MongoDB database already contains indexed YOLO documentation!
```
