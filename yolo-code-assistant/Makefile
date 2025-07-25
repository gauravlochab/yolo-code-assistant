# Makefile for YOLO Code Assistant
.PHONY: help install install-uv setup check index serve test test-simple test-unit test-integration test-all clean dev-install notebook dev run

# Default target
help:
	@echo "YOLO Code Assistant - Makefile Commands"
	@echo "======================================"
	@echo ""
	@echo "Setup Commands:"
	@echo "  make install-uv    - Install uv package manager"
	@echo "  make install       - Install dependencies using uv"
	@echo "  make dev-install   - Install with development dependencies"
	@echo "  make setup         - Complete setup (install uv, dependencies, and check)"
	@echo ""
	@echo "Application Commands:"
	@echo "  make check         - Check system configuration and connections"
	@echo "  make index         - Index the Ultralytics codebase"
	@echo "  make serve         - Launch the Gradio web interface"
	@echo "  make dev           - Quick development run (check + serve)"
	@echo "  make run           - Alias for serve"
	@echo ""
	@echo "Testing Commands:"
	@echo "  make test          - Run the simple test suite (recommended)"
	@echo "  make test-simple   - Run standalone test script"
	@echo "  make test-unit     - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make test-all      - Run all tests with pytest (if available)"
	@echo ""
	@echo "Development Commands:"
	@echo "  make notebook      - Start Jupyter notebook server"
	@echo "  make clean         - Clean generated files and cache"

# Install uv if not already installed
install-uv:
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "Please run 'source $$HOME/.local/bin/env' to add uv to PATH"; \
	else \
		echo "✅ uv is already installed"; \
	fi

# Install project dependencies
install:
	@if command -v uv >/dev/null 2>&1; then \
		echo "Installing dependencies with uv..."; \
		uv sync; \
		echo "✅ Dependencies installed successfully"; \
	else \
		echo "❌ Error: uv is not installed. Run 'make install-uv' first"; \
		exit 1; \
	fi

# Install with development dependencies
dev-install:
	@if command -v uv >/dev/null 2>&1; then \
		echo "Installing dependencies with dev extras..."; \
		uv sync --extra dev; \
		echo "✅ Development dependencies installed successfully"; \
	else \
		echo "❌ Error: uv is not installed. Run 'make install-uv' first"; \
		exit 1; \
	fi

# Complete setup
setup: install-uv
	@echo "🚀 Setting up YOLO Code Assistant..."
	@echo ""
	@if command -v uv >/dev/null 2>&1; then \
		make install; \
	else \
		echo "Please run 'source $$HOME/.local/bin/env' and then 'make install'"; \
		exit 1; \
	fi
	@echo ""
	@echo "✅ Setup complete! Next steps:"
	@echo "1. Copy .env.example to .env and add your credentials:"
	@echo "   cp .env.example .env"
	@echo "2. Edit .env file with your MongoDB URI and OpenRouter API key"
	@echo "3. Run 'make check' to verify configuration"
	@echo "4. Run 'make index' to index the codebase"
	@echo "5. Run 'make serve' to start the assistant"

# Check system configuration
check:
	@if [ ! -f .env ]; then \
		echo "❌ Error: .env file not found!"; \
		echo "Please copy .env.example to .env and add your credentials:"; \
		echo "  cp .env.example .env"; \
		exit 1; \
	fi
	@echo "🔍 Checking system configuration..."
	@uv run python main.py --check

# Index the Ultralytics codebase
index:
	@if [ ! -f .env ]; then \
		echo "❌ Error: .env file not found!"; \
		echo "Please copy .env.example to .env and add your credentials"; \
		exit 1; \
	fi
	@echo "📚 Indexing Ultralytics codebase..."
	@echo "This may take several minutes..."
	@uv run python main.py --index

# Launch the web interface
serve:
	@if [ ! -f .env ]; then \
		echo "❌ Error: .env file not found!"; \
		echo "Please copy .env.example to .env and add your credentials"; \
		exit 1; \
	fi
	@echo "🌐 Starting YOLO Code Assistant..."
	@echo "The interface will be available at http://localhost:7860"
	@echo "Press Ctrl+C to stop the server"
	@uv run python main.py --serve

# Testing Commands
# Default test command - runs the simple test suite
test: test-simple

# Run the standalone test script (recommended)
test-simple:
	@echo "🧪 Running YOLO Code Testing Suite..."
	@cd tests && uv run python simple_test.py

# Run unit tests only (requires pytest)
test-unit:
	@echo "🧪 Running unit tests..."
	@if command -v pytest >/dev/null 2>&1; then \
		cd tests && uv run python run_tests.py --type unit; \
	else \
		echo "⚠️  pytest not available, running simple test instead..."; \
		make test-simple; \
	fi

# Run integration tests only (requires pytest)
test-integration:
	@echo "🧪 Running integration tests..."
	@if command -v pytest >/dev/null 2>&1; then \
		cd tests && uv run python run_tests.py --type integration; \
	else \
		echo "⚠️  pytest not available, running simple test instead..."; \
		make test-simple; \
	fi

# Run all tests with pytest (if available)
test-all:
	@echo "🧪 Running all tests..."
	@if command -v pytest >/dev/null 2>&1; then \
		cd tests && uv run python run_tests.py --type all; \
	else \
		echo "⚠️  pytest not available, running simple test instead..."; \
		make test-simple; \
	fi

# Start Jupyter notebook server
notebook:
	@echo "📓 Starting Jupyter notebook server..."
	@echo "Notebooks will be available at http://localhost:8888"
	@cd notebooks && uv run jupyter notebook

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	@rm -rf .pytest_cache 2>/dev/null || true
	@rm -rf .ruff_cache 2>/dev/null || true
	@rm -rf .coverage 2>/dev/null || true
	@rm -rf htmlcov 2>/dev/null || true
	@echo "✅ Clean complete"

# Development shortcuts

# Quick development run (check + serve)
dev: 
	@echo "🚀 Starting development mode..."
	@make check && make serve

# Alias for serve
run: serve

# Validate the testing framework
validate-tests:
	@echo "🔍 Validating testing framework..."
	@make test-simple
	@echo "✅ Testing framework validation complete"

# Show project status
status:
	@echo "📊 YOLO Code Assistant Status"
	@echo "============================="
	@echo ""
	@if [ -f .env ]; then \
		echo "✅ Environment file: .env exists"; \
	else \
		echo "❌ Environment file: .env missing"; \
	fi
	@if command -v uv >/dev/null 2>&1; then \
		echo "✅ Package manager: uv installed"; \
	else \
		echo "❌ Package manager: uv not found"; \
	fi
	@if [ -d data/ultralytics_repo ]; then \
		echo "✅ Codebase: Ultralytics repo cloned"; \
	else \
		echo "⚠️  Codebase: Ultralytics repo not found (run 'make index')"; \
	fi
	@echo ""
	@echo "Run 'make help' for available commands"
