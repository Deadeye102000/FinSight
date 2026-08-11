"""Scaffold verification tests for FinSight."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRECTORIES = [
    ROOT / "finsight",
    ROOT / "finsight" / "mcp_server",
    ROOT / "finsight" / "mcp_server" / "tools",
    ROOT / "finsight" / "mcp_server" / "utils",
    ROOT / "finsight" / "agent",
    ROOT / "finsight" / "api",
    ROOT / "finsight" / "ui",
    ROOT / "tests",
]

REQUIRED_FILES = [
    ROOT / "finsight" / "__init__.py",
    ROOT / "finsight" / "mcp_server" / "__init__.py",
    ROOT / "finsight" / "mcp_server" / "server.py",
    ROOT / "finsight" / "mcp_server" / "tools" / "__init__.py",
    ROOT / "finsight" / "mcp_server" / "tools" / "price.py",
    ROOT / "finsight" / "mcp_server" / "tools" / "fundamentals.py",
    ROOT / "finsight" / "mcp_server" / "tools" / "sentiment.py",
    ROOT / "finsight" / "mcp_server" / "tools" / "announcements.py",
    ROOT / "finsight" / "mcp_server" / "tools" / "peers.py",
    ROOT / "finsight" / "mcp_server" / "utils" / "__init__.py",
    ROOT / "finsight" / "mcp_server" / "utils" / "validators.py",
    ROOT / "finsight" / "agent" / "__init__.py",
    ROOT / "finsight" / "agent" / "orchestrator.py",
    ROOT / "finsight" / "api" / "__init__.py",
    ROOT / "finsight" / "api" / "main.py",
    ROOT / "finsight" / "ui" / "app.py",
    ROOT / "tests" / "__init__.py",
    ROOT / "tests" / "test_price.py",
    ROOT / "tests" / "test_fundamentals.py",
    ROOT / "tests" / "test_sentiment.py",
    ROOT / "tests" / "test_announcements.py",
    ROOT / "tests" / "test_peers.py",
    ROOT / "tests" / "test_api.py",
    ROOT / "tests" / "test_scaffold.py",
    ROOT / ".env.example",
    ROOT / ".env",
    ROOT / ".gitignore",
    ROOT / "requirements.txt",
    ROOT / "README.md",
    ROOT / "pyproject.toml",
]

REQUIRED_ENV_KEYS = {
    "ANTHROPIC_API_KEY",
    "NEWS_API_KEY",
    "MCP_SERVER_HOST",
    "MCP_SERVER_PORT",
    "API_HOST",
    "API_PORT",
}

REQUIRED_PACKAGES = {
    "mcp>=1.0.0",
    "anthropic>=0.40.0",
    "yfinance>=0.2.40",
    "fastapi>=0.115.0",
    "uvicorn>=0.32.0",
    "streamlit>=1.40.0",
    "httpx>=0.27.0",
    "python-dotenv>=1.0.0",
    "transformers>=4.40.0",
    "torch>=2.0.0",
    "pydantic>=2.0.0",
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "pytest-mock>=3.14.0",
    "requests>=2.32.0",
    "pandas>=2.0.0",
    "numpy>=1.26.0",
}


def test_required_files_exist() -> None:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED_FILES if not path.is_file()]
    assert not missing, f"Missing files: {missing}"


def test_required_directories_exist() -> None:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED_DIRECTORIES if not path.is_dir()]
    assert not missing, f"Missing directories: {missing}"


def test_env_example_contains_required_keys() -> None:
    env_example = (ROOT / ".env.example").read_text()
    missing = [key for key in REQUIRED_ENV_KEYS if key not in env_example]
    assert not missing, f"Missing .env.example keys: {missing}"


def test_requirements_contains_required_packages() -> None:
    requirements = (ROOT / "requirements.txt").read_text()
    missing = [package for package in REQUIRED_PACKAGES if package not in requirements]
    assert not missing, f"Missing requirements packages: {missing}"


def test_env_is_gitignored() -> None:
    gitignore = (ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in gitignore, ".env must be listed in .gitignore"
