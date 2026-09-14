from pathlib import Path


def test_pyproject_project_contract_exists():
    root = Path(__file__).resolve().parents[1]
    pyproject = root / "pyproject.toml"

    assert pyproject.exists(), "pyproject.toml metadata contract is missing"
    data = pyproject.read_text(encoding="utf-8")

    assert "[project]" in data
    assert 'name = "afm-marker-recognition"' in data
    assert 'requires-python = ">=3.9"' in data
    assert '[project.scripts]' in data
    assert 'afmpipeline = "main:run_cli"' in data


def test_requirements_file_exists_for_runtime_dependencies():
    root = Path(__file__).resolve().parents[1]
    requirements = root / "requirements.txt"

    assert requirements.exists(), "requirements.txt runtime metadata is missing"
    requirements_text = requirements.read_text(encoding="utf-8")

    assert "numpy" in requirements_text
    assert "tensorflow" in requirements_text
    assert "scikit-image" in requirements_text
