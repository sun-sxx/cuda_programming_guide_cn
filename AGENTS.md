# AGENTS.md - CUDA Programming Guide CN

This document provides guidelines for AI coding agents working in this repository.

## Project Overview

This is a Sphinx documentation project for translating the CUDA Programming Guide to Chinese (zh_CN). The project uses reStructuredText (RST) format for documentation.

**Original Documentation**: https://docs.nvidia.com/cuda/cuda-programming-guide/index.html

## Dependencies

```bash
pip install -r requirements.txt        # Install Python dependencies
sudo apt-get install texlive-full      # For PDF generation (Ubuntu/Debian)
```

Core packages: `sphinx`, `sphinx-rtd-theme`, `rstcheck`, `doc8`, `sphinx-autobuild`

## Build Commands

```bash
make html                           # Build HTML documentation
make latexpdf                       # Build PDF documentation
make html latexpdf                  # Build all formats
make help                           # View all build targets
make clean                          # Clean build directory
sphinx-build -M html source build   # Direct sphinx-build command
sphinx-autobuild source build/html  # Live reload during development
```

## Lint and Validation Commands

```bash
rstcheck source/                          # Check RST syntax
doc8 source/                              # Check with doc8
sphinx-build -M html source build -W      # Validate (treat warnings as errors)
sphinx-build -M linkcheck source build    # Check for broken links
```

## Test Commands

This is a documentation project without traditional unit tests. Verify with:

```bash
sphinx-build -M html source build -W      # Build must succeed without warnings
rstcheck source/ && doc8 source/ && sphinx-build -M html source build -W  # Full validation
```

## Project Structure

```
cuda_guide_cn/
├── source/                    # Source files (conf.py, index.rst, _static/, _templates/)
├── build/                     # Generated documentation (gitignored)
├── Makefile                   # Build commands (Unix/Linux/macOS)
├── make.bat                   # Build commands (Windows)
└── requirements.txt          # Python dependencies
```

## Code Style Guidelines

### reStructuredText (RST) Files

**Headings** - Use consistent hierarchy: `=` (title), `-` (section), `^` (subsection), `~` (sub-subsection)

**Directives**:
```rst
.. directive-name:: argument
   :option1: value1

   Content with proper indentation (3 spaces).
```

**Code Blocks**:
```rst
.. code-block:: cuda
   :caption: Example kernel
   :linenos:

   __global__ void kernel() {
       // CUDA code here
   }
```

**Cross-references**:
```rst
:doc:`document-name`          # Link to another document
:ref:`label-name`             # Link to a labeled section

.. _introduction:         # Define a label

Introduction
------------
This section can be referenced via :ref:`sec:introduction`.
```

**Lists**:
```rst
- Bullet item 1
- Bullet item 2

1. Numbered item 1
2. Numbered item 2
```

**Admonitions**:
```rst
.. note:: Important note here.
.. warning:: Warning message here.
.. tip:: Helpful tip here.
```

### Python Files (conf.py)

```python
# Standard library imports first (alphabetical)
import os
import sys

# Third-party imports second (alphabetical)
from pathlib import Path

# Constants in UPPER_CASE
PROJECT_NAME = 'CUDA Programming Guide CN'

# Configuration
project = 'cuda rogramming guide cn'
copyright = '2026, Michael'
author = 'Michael'
release = '0.1'
```

## Chinese Localization Guidelines

- **Punctuation**: Use Chinese punctuation marks: ，。！？；：
- **Quotation marks**: Use Chinese quotation marks: 「」 or ""
- **Spaces**: Add spaces between Chinese and English/numbers
- **Example**: `CUDA 是 NVIDIA 开发的并行计算平台，支持 C++ 和 Python。`
- **Translation**: Translate prose to Chinese but keep code in English
- **Technical terms**: Keep original English terms for API names, function names

## Naming Conventions

**RST File Names**: Use lowercase with hyphens (`programming-model.rst`), keep names short but descriptive, group related files with prefixes (`memory-*.rst`)

**RST Labels**: Use descriptive prefixes (`sec:`, `fig:`, `tbl:`, `lst:`, `eq:`), e.g., `.. _programming-model:`

## Error Handling

- **Treat warnings as errors**: Always use `-W` flag during development
- **Fix broken references immediately**: Broken refs cause build failures
- **Check external links periodically**: Use `sphinx-build -M linkcheck`
- **Handle encoding**: Ensure UTF-8 encoding for Chinese characters
- **RST syntax errors**: Use `rstcheck` to find syntax issues before building

## Best Practices

1. **Consistent Formatting**: Maintain consistent RST formatting across all files
2. **Line Length**: Keep lines under 120 characters for readability
3. **Blank Lines**: Use blank lines to separate sections and directives
4. **Indentation**: Use 3 spaces for RST content indentation (not tabs)
5. **Version Control**: Commit source files only, not build artifacts
6. **Build Verification**: Always verify build succeeds after changes

## Common Tasks

**Add New Document**: Create `.rst` file in `source/`, add to `toctree` in `index.rst`, build and verify: `make html`

**Update Configuration**: Edit `source/conf.py` for theme changes, extensions, custom settings

**Check for Issues**: `rstcheck source/ && doc8 source/ && sphinx-build -M html source build -W --keep-going`

**Fix Broken References**: `sphinx-build -M html source build -W 2>&1 | grep -i "reference"`