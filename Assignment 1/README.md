# "Sapien" - Search Engine Project

IMPORTANT: Each group should pick a name for its project. "sapien" is just a template.

This project implements a search engine called "Sapien" with both a REST API for searching and a CLI interface for indexing. It's built with strict typing, linting, and memory constraints to ensure high-quality, efficient code.

## Prerequisites

- uv (install from https://docs.astral.sh/uv/getting-started/installation/)

## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```

2. You don't need to activate the environment manually. Just let `uv` handle the activation automatically by prefixing your commands with `uv run`.

3. Install pre-commit hooks (runs linting on every commit):
   ```bash
   uv run pre-commit install
   ```

## Project Organization

The project is organized into several key components:

```
src/sapien/
├── core/                    # Core models and utilities
│   ├── model.py            # Document data models (Pydantic)
│   └── limit_memory.py     # Memory monitoring utilities
├── entrypoints/            # Application entry points
│   ├── api/               # FastAPI REST API
│   │   ├── app.py        # FastAPI application setup
│   │   ├── model.py      # API request/response models
│   │   └── routes/       # API route handlers
│   │       ├── search.py      # Search endpoints
│   │       └── healthcheck.py # Health check endpoint
│   ├── cli.py            # Command-line indexer interface
│   └── asgi.py           # ASGI server entry point
static_pages/             # static webpage that interacts with your search engine via REST
```

### Key Entrypoints:

- **REST API**: FastAPI-based search interface accessible at `/api/v1/search/`
- **CLI Indexer**: Command-line tool for building search indices with memory limiting (≤2GB)

## Usage

### Running the Search API

Start the FastAPI server:
```bash
uv run uvicorn sapien.entrypoints.asgi:app --reload
```
- **Static Web Interface**: `http://localhost:8000`
- **API docs**: `http://localhost:8000/docs`

### Running the CLI Indexer

The CLI indexer runs with memory monitoring enabled to enforce the 2GB memory limit:
```bash
uv run cli [arguments]
```
OR
```bash
uv run src/sapien/entrypoints/cli [arguments]
```

**Note**: The CLI automatically starts memory monitoring, which must be included as part of the assignment.
Each group must be able to develop an indexer with memory constraints.

## Development Best Practices

### Code Quality

This project enforces strict code quality standards:

- **Type Checking**: Full type annotation coverage with Pyright
- **Linting**: Ruff with comprehensive rule set (100-char line limit)
- **Formatting**: Automatic code formatting with Ruff
- **Import Sorting**: Organized imports with isort integration

### Git Workflow

1. **Make frequent commits**: Pre-commit hooks run type checking and linting automatically
2. **Follow the enforced standards**: The project uses strict linting rules for consistency
3. **Test before committing**: All code is validated before entering the repository

### Skipping Pre-commit Hooks

⚠️ **NOT RECOMMENDED!**

This repository includes pre-commit hooks that verify and standardize code before committing. However, if you're close to a deadline and need to bypass them temporarily:

```bash
git commit --no-verify -m "Your commit message"
```
