"""
Tests for .gitignore contents and structure.

Testing library/framework: pytest
- We use plain pytest-style tests and assertions.
- Tests validate the presence of critical custom patterns from the PR diff,
  ensure no duplicates among active patterns, and check formatting hygiene.

Strategy:
- Keep an embedded snapshot (GITIGNORE_CONTENT) of the intended .gitignore as provided in the diff.
- If a project root .gitignore exists, tests validate that it contains all critical patterns and
  matches key hygiene constraints. If not present, tests still validate the embedded snapshot so
  CI provides actionable signals without hard failing on missing file.

Notes:
- "Active patterns" exclude blank lines and comments (# ...).
- We specifically check custom, non-standard additions from the diff: Dataset/, Info/, Generation-Result/,
  ALLaM-Models/, environment.yml, Submission-last.zip, marimo/*, Cursor files, Abstra dir, Ruff cache, etc.
"""

from __future__ import annotations
import os
import re
from typing import List, Set


# Embedded snapshot of the .gitignore content from the diff
GITIGNORE_CONTENT = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[codz]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
Dataset/
Info/
Generation-Result/
src/
ALLaM-Models/
environment.yml
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py.cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy


# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
#   For a library or package, you might want to ignore these files since the code is
#   intended to run in multiple environments; otherwise, check them in:
# .python-version

# pipenv
#   According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.
#   However, in case of collaboration, if having platform-specific dependencies or dependencies
#   having no cross-platform support, pipenv may install dependencies that don't work, or not
#   install all needed dependencies.
#Pipfile.lock

# UV
#   Similar to Pipfile.lock, it is generally recommended to include uv.lock in version control.
#   This is especially recommended for binary packages to ensure reproducibility, and is more
#   commonly ignored for libraries.
#uv.lock

# poetry
#   Similar to Pipfile.lock, it is generally recommended to include poetry.lock in version control.
#   This is especially recommended for binary packages to ensure reproducibility, and is more
#   commonly ignored for libraries.
#   https://python-poetry.org/docs/basic-usage/#commit-your-poetrylock-file-to-version-control
#poetry.lock
#poetry.toml

# pdm
#   Similar to Pipfile.lock, it is generally recommended to include pdm.lock in version control.
#   pdm recommends including project-wide configuration in pdm.toml, but excluding .pdm-python.
#   https://pdm-project.org/en/latest/usage/project/#working-with-version-control
#pdm.lock
#pdm.toml
.pdm-python
.pdm-build/

# pixi
#   Similar to Pipfile.lock, it is generally recommended to include pixi.lock in version control.
#pixi.lock
#   Pixi creates a virtual environment in the .pixi directory, just like venv module creates one
#   in the .venv directory. It is recommended not to include this directory in version control.
.pixi

# PEP 582; used by e.g. github.com/David-OConnor/pyflow and github.com/pdm-project/pdm
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.envrc
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site
 
# mypy
.mypy_cache/
.dmypy.json
dmypy.json


# Pyre type checker
.pyre/
Submission-last.zip
# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# PyCharm
#  JetBrains specific template is maintained in a separate JetBrains.gitignore that can
#  be found at https://github.com/github/gitignore/blob/main/Global/JetBrains.gitignore
#  and can be added to the global gitignore or merged into this file.  For a more nuclear
#  option (not recommended) you can uncomment the following to ignore the entire idea folder.
#.idea/

# Abstra
# Abstra is an AI-powered process automation framework.
# Ignore directories containing user credentials, local state, and settings.
# Learn more at https://abstra.io/docs
.abstra/

# Visual Studio Code
#  Visual Studio Code specific template is maintained in a separate VisualStudioCode.gitignore 
#  that can be found at https://github.com/github/gitignore/blob/main/Global/VisualStudioCode.gitignore
#  and can be added to the global gitignore or merged into this file. However, if you prefer, 
#  you could uncomment the following to ignore the entire vscode folder
# .vscode/

# Ruff stuff:
.ruff_cache/

# PyPI configuration file
.pypirc

# Cursor
#  Cursor is an AI-powered code editor. `.cursorignore` specifies files/directories to
#  exclude from AI features like autocomplete and code analysis. Recommended for sensitive data
#  refer to https://docs.cursor.com/context/ignore-files
.cursorignore
.cursorindexingignore

# Marimo
marimo/_static/
marimo/_lsp/
__marimo__/
"""

def _read_repo_gitignore() -> str | None:
    """Return the content of the project's root .gitignore if it exists, else None."""
    path = os.path.join(os.getcwd(), ".gitignore")
    if os.path.exists(path) and os.path.isfile(path):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    return None

def _active_patterns(content: str) -> List[str]:
    """Return active ignore patterns (non-empty, non-comment), stripped of surrounding whitespace."""
    lines = [ln.rstrip("\n") for ln in content.splitlines()]
    active: List[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        active.append(s)
    return active

def _raw_lines(content: str) -> List[str]:
    return [ln.rstrip("\n") for ln in content.splitlines()]

# Critical custom patterns primarily from the PR diff (non-standard additions worth validating)
CRITICAL_CUSTOM_PATTERNS: List[str] = [
    "Dataset/",
    "Info/",
    "Generation-Result/",
    "src/",
    "ALLaM-Models/",
    "environment.yml",
    "Submission-last.zip",
    ".pypirc",
    ".ruff_cache/",
    ".abstra/",
    ".cursorignore",
    ".cursorindexingignore",
    "marimo/_static/",
    "marimo/_lsp/",
    "__marimo__/",
]

# Commented lockfiles we expect to remain commented (ensuring they are NOT active ignores)
COMMENTED_LOCKFILES: List[str] = [
    "Pipfile.lock",
    "uv.lock",
    "poetry.lock",
    "poetry.toml",
    "pdm.lock",
    "pdm.toml",
    "pixi.lock",
    ".vscode/",
    ".idea/",
    ".python-version",
]

# A few canonical Python patterns that should nearly always be present
PYTHON_CANONICALS: List[str] = [
    "__pycache__/",
    "*.py[codz]",
    "*$py.class",
    ".venv",
    "env/",
    "venv/",
    ".pytest_cache/",
    ".mypy_cache/",
    "build/",
    "dist/",
    "*.egg-info/",
]


def test_embedded_snapshot_ends_with_newline():
    # Hygiene: ensure newline at EOF in the snapshot (important for file diffs)
    assert GITIGNORE_CONTENT.endswith("\n"), "Embedded .gitignore snapshot should end with a newline"


def test_no_crlf_in_snapshot():
    # Hygiene: disallow CRLF line endings in snapshot
    assert "\r\n" not in GITIGNORE_CONTENT, "Snapshot should use LF line endings only"


def test_no_trailing_whitespace_in_snapshot():
    offenders = [i + 1 for i, ln in enumerate(_raw_lines(GITIGNORE_CONTENT)) if re.search(r"[ \t]+$", ln)]
    assert offenders == [], f"Lines with trailing whitespace: {offenders}"


def test_no_duplicate_active_patterns_in_snapshot():
    patterns = _active_patterns(GITIGNORE_CONTENT)
    seen: Set[str] = set()
    dups: List[str] = []
    for p in patterns:
        if p in seen:
            dups.append(p)
        else:
            seen.add(p)
    assert dups == [], f"Duplicate active patterns found: {dups}"


def test_snapshot_contains_python_canonicals():
    active = set(_active_patterns(GITIGNORE_CONTENT))
    missing = [p for p in PYTHON_CANONICALS if p not in active]
    assert missing == [], f"Snapshot missing canonical Python patterns: {missing}"


def test_snapshot_contains_critical_customs():
    active = set(_active_patterns(GITIGNORE_CONTENT))
    missing = [p for p in CRITICAL_CUSTOM_PATTERNS if p not in active]
    assert missing == [], f"Snapshot missing critical custom patterns: {missing}"


def test_snapshot_lockfiles_are_commented_not_active():
    active = set(_active_patterns(GITIGNORE_CONTENT))
    raw = "\n" + GITIGNORE_CONTENT  # simplify anchored searches
    mistakenly_active = [lf for lf in COMMENTED_LOCKFILES if lf in active]
    assert mistakenly_active == [], f"Lockfiles should be commented out, but active: {mistakenly_active}"
    # Ensure commented lines actually present
    missing_comments = []
    for lf in COMMENTED_LOCKFILES:
        pat = f"\n#{lf}"
        if pat not in raw:
            missing_comments.append(lf)
    assert missing_comments == [], f"Expected commented entries not present: {missing_comments}"


def test_snapshot_does_not_glob_everything():
    # Ensure no raw '*' line that would ignore everything
    assert "\n*\n" not in ("\n" + GITIGNORE_CONTENT + "\n"), "A bare '*' pattern would ignore everything"


def test_project_gitignore_includes_critical_patterns_or_skip(monkeypatch):
    """
    If the project root .gitignore exists, ensure it includes the critical custom patterns.
    If it doesn't exist, skip gracefully but still validate the embedded snapshot via other tests.
    """
    content = _read_repo_gitignore()
    if content is None:
        import pytest
        pytest.skip("Project .gitignore not found; validating embedded snapshot only.")
    active = set(_active_patterns(content))
    missing = [p for p in CRITICAL_CUSTOM_PATTERNS if p not in active]
    assert missing == [], f"Project .gitignore missing critical custom patterns: {missing}"


def test_project_gitignore_has_no_duplicate_active_patterns_or_skip():
    content = _read_repo_gitignore()
    if content is None:
        import pytest
        pytest.skip("Project .gitignore not found; validating embedded snapshot only.")
    patterns = _active_patterns(content)
    seen: Set[str] = set()
    dups: List[str] = []
    for p in patterns:
        if p in seen:
            dups.append(p)
        else:
            seen.add(p)
    assert dups == [], f"Project .gitignore has duplicate active patterns: {dups}"


def test_project_gitignore_lockfiles_are_comment_only_or_skip():
    content = _read_repo_gitignore()
    if content is None:
        import pytest
        pytest.skip("Project .gitignore not found; validating embedded snapshot only.")
    active = set(_active_patterns(content))
    mistakenly_active = [lf for lf in COMMENTED_LOCKFILES if lf in active]
    assert mistakenly_active == [], f"Project .gitignore should not actively ignore these lockfiles: {mistakenly_active}"