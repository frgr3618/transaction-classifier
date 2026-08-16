import os
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture(scope="session")
def app_module():
    """Import src/app.py.

    app.py loads model.joblib and tfidf.joblib by relative path, so the working
    directory has to be src/ at import time — that is what the Dockerfile does.
    """
    cwd = os.getcwd()
    os.chdir(SRC)
    sys.path.insert(0, str(SRC))
    try:
        import app

        yield app
    finally:
        sys.path.remove(str(SRC))
        os.chdir(cwd)


@pytest.fixture(scope="session")
def client(app_module):
    from fastapi.testclient import TestClient

    return TestClient(app_module.app)


CATEGORIES = {
    "education",
    "emi",
    "entertainment",
    "food",
    "healthcare",
    "investment",
    "shopping",
    "travel",
    "utilities",
}
