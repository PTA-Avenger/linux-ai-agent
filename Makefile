.PHONY: help install install-dev test test-basic test-ai lint format clean build docs
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Linux AI Agent - Development Commands"
	@echo "====================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package and basic dependencies
	pip install -e .

install-dev: ## Install package with development dependencies
	pip install -e ".[dev]"

install-ai: ## Install package with AI dependencies
	pip install -e ".[ai,enhanced]"

install-all: ## Install package with all dependencies
	pip install -e ".[ai,enhanced,dev]"

test: ## Run all tests
	python -m pytest tests/ -v

test-basic: ## Run basic functionality tests
	python test_basic.py

test-ai: ## Run AI-specific tests (requires AI dependencies)
	python test_ai_fixes.py

test-improvements: ## Run improvement tests
	python test_improvements.py

lint: ## Run code linting
	flake8 src/ tests/
	mypy src/ --ignore-missing-imports

format: ## Format code with black
	black src/ tests/ *.py
	isort src/ tests/ *.py

format-check: ## Check code formatting without making changes
	black --check src/ tests/ *.py
	isort --check-only src/ tests/ *.py

demo: ## Run the demo script
	python demo.py

run: ## Start the interactive CLI
	python src/main.py

clean: ## Clean up build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

build: ## Build distribution packages
	python -m build

docs: ## Generate documentation (placeholder)
	@echo "Documentation generation not implemented yet"
	@echo "Consider adding sphinx or mkdocs"

security-scan: ## Run security scanning
	bandit -r src/

dependency-check: ## Check for dependency vulnerabilities
	safety check

setup-venv: ## Set up virtual environment
	python -m venv venv
	@echo "Activate with: source venv/bin/activate"

check-clamav: ## Check if ClamAV is installed
	@which clamscan >/dev/null 2>&1 && echo "✅ ClamAV installed" || echo "❌ ClamAV not found. Install with: sudo apt install clamav"

install-system-deps: ## Install system dependencies (requires sudo)
	sudo apt update
	sudo apt install -y clamav clamav-daemon python3-venv

update-clamav: ## Update ClamAV virus definitions
	sudo freshclam

status: ## Show project status
	@echo "=== Linux AI Agent Status ==="
	@echo "Python version: $(shell python --version)"
	@echo "Pip version: $(shell pip --version)"
	@echo "Files: $(shell find src -name '*.py' | wc -l) Python files"
	@echo "Lines: $(shell find src -name '*.py' -exec wc -l {} + | tail -1)"
	@echo "Tests: $(shell find . -maxdepth 1 -name 'test_*.py' | wc -l) test files"
	@echo "Docs: $(shell find . -maxdepth 1 -name '*.md' | wc -l) documentation files"
	@make check-clamav