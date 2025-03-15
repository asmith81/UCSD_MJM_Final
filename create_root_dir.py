"""
Setup script to create the directory structure for the invoice processing project.
This script will create the structure in the current directory without creating a new root folder.
"""

import os
import json
import yaml
from pathlib import Path
import shutil

def create_directory(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {path}")

def create_file(path, content=""):
    """Create an empty file or with minimal content."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
    print(f"Created file: {path}")

def create_python_file(path, module_name=None):
    """Create a Python file with minimal imports and documentation."""
    if module_name is None:
        module_name = Path(path).stem
    
    content = f'''"""
{module_name} module for Invoice Processing project.
"""

# Standard library imports

# Third-party imports

# Local imports

'''
    create_file(path, content)

def create_init_file(path, module_docstring=None):
    """Create a Python __init__.py file."""
    if module_docstring is None:
        module_name = Path(path).parent.name
        module_docstring = f"{module_name} module initialization"
    
    content = f'''"""
{module_docstring}
"""

'''
    create_file(path, content)

def create_yaml_file(path, sample_content=None):
    """Create a YAML configuration file with sample content."""
    if sample_content is None:
        sample_content = {
            "name": Path(path).stem,
            "description": "Sample configuration",
            "parameters": {
                "param1": "value1",
                "param2": "value2"
            }
        }
    
    with open(path, 'w') as f:
        yaml.dump(sample_content, f, default_flow_style=False)
    
    print(f"Created YAML file: {path}")

def create_notebook(path, title=None):
    """Create a minimal Jupyter notebook file."""
    if title is None:
        title = Path(path).stem.replace('_', ' ').title()
    
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# {title}\n\n## Overview\n\nDescription of this notebook's purpose."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Import dependencies\nimport os\nimport sys\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\n\n# Add project root to path\nsys.path.append('../')\n\n# Import project modules\nfrom src.config.paths import PathConfig"]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(path, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"Created notebook: {path}")

def create_markdown_file(path, title=None):
    """Create a Markdown file with basic content."""
    if title is None:
        title = Path(path).stem.replace('_', ' ').title()
    
    content = f"""# {title}

## Overview

Description of this document.

## Details

Additional details will go here.

## Examples

Examples will go here.
"""
    
    create_file(path, content)

def create_shell_script(path, description=None):
    """Create a shell script with basic content."""
    if description is None:
        description = f"Script to {Path(path).stem.replace('_', ' ')}"
    
    content = f"""#!/bin/bash
# {description}

echo "Running {Path(path).name}..."

# Your script content goes here

echo "Completed {Path(path).name}"
"""
    
    create_file(path, content)
    # Make the script executable
    os.chmod(path, 0o755)

def setup_project_in_current_dir():
    """Set up the complete project structure in the current directory."""
    # Create configs structure
    configs_dir = "configs"
    
    # Environment configs
    create_directory(os.path.join(configs_dir, "environments"))
    create_yaml_file(os.path.join(configs_dir, "environments", "local.yaml"))
    create_yaml_file(os.path.join(configs_dir, "environments", "runpod.yaml"))
    
    # Experiment configs
    create_directory(os.path.join(configs_dir, "experiments"))
    create_yaml_file(os.path.join(configs_dir, "experiments", "baseline.yaml"))
    create_yaml_file(os.path.join(configs_dir, "experiments", "prompt_comparison.yaml"))
    create_yaml_file(os.path.join(configs_dir, "experiments", "model_comparison.yaml"))
    
    # Model configs
    create_directory(os.path.join(configs_dir, "models"))
    create_yaml_file(os.path.join(configs_dir, "models", "pixtral-12b.yaml"))
    create_yaml_file(os.path.join(configs_dir, "models", "future_models.yaml"))
    
    # Prompt configs
    create_directory(os.path.join(configs_dir, "prompts"))
    create_yaml_file(os.path.join(configs_dir, "prompts", "basic.yaml"))
    create_yaml_file(os.path.join(configs_dir, "prompts", "detailed.yaml"))
    create_yaml_file(os.path.join(configs_dir, "prompts", "positioned.yaml"))
    
    # Ensure data structure exists
    data_dir = "data"
    create_directory(data_dir)
    create_directory(os.path.join(data_dir, "images"))
    
    # Create models structure
    create_directory(os.path.join("models", "cache"))
    
    # Create notebooks structure
    notebooks_dir = "notebooks"
    create_directory(notebooks_dir)
    create_notebook(os.path.join(notebooks_dir, "01_environment_setup.ipynb"), "Environment Setup and Validation")
    create_notebook(os.path.join(notebooks_dir, "02_single_model_test.ipynb"), "Single Model Testing")
    create_notebook(os.path.join(notebooks_dir, "03_prompt_comparison.ipynb"), "Prompt Comparison Analysis")
    create_notebook(os.path.join(notebooks_dir, "04_model_comparison.ipynb"), "Model Comparison Analysis")
    create_notebook(os.path.join(notebooks_dir, "05_full_experiment_grid.ipynb"), "Full Experiment Grid Execution")
    create_notebook(os.path.join(notebooks_dir, "06_results_analysis.ipynb"), "Results Analysis and Visualization")
    
    # Create utilities notebooks
    utilities_dir = os.path.join(notebooks_dir, "utilities")
    create_directory(utilities_dir)
    create_notebook(os.path.join(utilities_dir, "data_exploration.ipynb"), "Data Exploration")
    create_notebook(os.path.join(utilities_dir, "gpu_benchmarking.ipynb"), "GPU Benchmarking")
    create_notebook(os.path.join(utilities_dir, "error_analysis.ipynb"), "Error Analysis")
    
    # Create results structure
    results_dir = "results"
    
    # Raw results
    create_directory(os.path.join(results_dir, "raw"))
    # (Timestamped directories will be created during execution)
    
    # Processed results
    create_directory(os.path.join(results_dir, "processed", "model_comparisons"))
    create_directory(os.path.join(results_dir, "processed", "prompt_comparisons"))
    create_directory(os.path.join(results_dir, "processed", "trend_analysis"))
    
    # Visualizations
    create_directory(os.path.join(results_dir, "visualizations", "accuracy_charts"))
    create_directory(os.path.join(results_dir, "visualizations", "error_analysis"))
    create_directory(os.path.join(results_dir, "visualizations", "performance_charts"))
    
    # Create src structure
    src_dir = "src"
    create_init_file(os.path.join(src_dir, "__init__.py"), "Invoice Processing source module")
    
    # Config module
    config_dir = os.path.join(src_dir, "config")
    create_directory(config_dir)
    create_init_file(os.path.join(config_dir, "__init__.py"))
    create_python_file(os.path.join(config_dir, "experiment.py"))
    create_python_file(os.path.join(config_dir, "paths.py"))
    
    # Data module
    data_module_dir = os.path.join(src_dir, "data")
    create_directory(data_module_dir)
    create_init_file(os.path.join(data_module_dir, "__init__.py"))
    create_python_file(os.path.join(data_module_dir, "loader.py"))
    create_python_file(os.path.join(data_module_dir, "preprocessor.py"))
    
    # Models module
    models_dir = os.path.join(src_dir, "models")
    create_directory(models_dir)
    create_init_file(os.path.join(models_dir, "__init__.py"))
    create_python_file(os.path.join(models_dir, "loader.py"))
    create_python_file(os.path.join(models_dir, "model_configs.py"))
    create_python_file(os.path.join(models_dir, "model_utils.py"))
    create_python_file(os.path.join(models_dir, "optimization.py"))
    create_python_file(os.path.join(models_dir, "registry.py"))
    
    # Prompts module
    prompts_dir = os.path.join(src_dir, "prompts")
    create_directory(prompts_dir)
    create_init_file(os.path.join(prompts_dir, "__init__.py"))
    create_python_file(os.path.join(prompts_dir, "invoice_prompts.py"))
    create_python_file(os.path.join(prompts_dir, "prompt_utils.py"))
    create_python_file(os.path.join(prompts_dir, "registry.py"))
    
    # Execution module
    execution_dir = os.path.join(src_dir, "execution")
    create_directory(execution_dir)
    create_init_file(os.path.join(execution_dir, "__init__.py"))
    create_python_file(os.path.join(execution_dir, "batch.py"))
    create_python_file(os.path.join(execution_dir, "inference.py"))
    create_python_file(os.path.join(execution_dir, "pipeline.py"))
    
    # Results module
    results_module_dir = os.path.join(src_dir, "results")
    create_directory(results_module_dir)
    create_init_file(os.path.join(results_module_dir, "__init__.py"))
    create_python_file(os.path.join(results_module_dir, "collector.py"))
    create_python_file(os.path.join(results_module_dir, "schema.py"))
    create_python_file(os.path.join(results_module_dir, "storage.py"))
    
    # Analysis module
    analysis_dir = os.path.join(src_dir, "analysis")
    create_directory(analysis_dir)
    create_init_file(os.path.join(analysis_dir, "__init__.py"))
    create_python_file(os.path.join(analysis_dir, "metrics.py"))
    create_python_file(os.path.join(analysis_dir, "statistics.py"))
    create_python_file(os.path.join(analysis_dir, "visualization.py"))
    
    # Create scripts
    scripts_dir = "scripts"
    create_directory(scripts_dir)
    create_shell_script(os.path.join(scripts_dir, "setup_runpod.sh"), "Setup script for RunPod environment")
    create_python_file(os.path.join(scripts_dir, "run_experiment.py"))
    create_python_file(os.path.join(scripts_dir, "aggregate_results.py"))
    create_python_file(os.path.join(scripts_dir, "generate_report.py"))
    
    # Create docs
    docs_dir = "docs"
    create_directory(docs_dir)
    create_markdown_file(os.path.join(docs_dir, "environment_setup.md"), "Environment Setup Guide")
    create_markdown_file(os.path.join(docs_dir, "experiment_configuration.md"), "Experiment Configuration Guide")
    create_markdown_file(os.path.join(docs_dir, "model_registry.md"), "Model Registry Documentation")
    create_markdown_file(os.path.join(docs_dir, "result_interpretation.md"), "Result Interpretation Guide")
    
    # Create or update root files (only if they don't exist)
    if not os.path.exists("requirements.txt"):
        create_file("requirements.txt", 
                "# Project dependencies\n"
                "torch>=2.0.0\n"
                "transformers>=4.34.0\n"
                "accelerate>=0.21.0\n"
                "sentencepiece>=0.1.99\n"
                "tqdm>=4.66.1\n"
                "pillow>=10.0.0\n"
                "matplotlib>=3.7.2\n"
                "pandas>=2.0.3\n"
                "python-Levenshtein>=0.21.1\n"
                "bitsandbytes>=0.41.1\n"
                "numpy>=1.24.3\n"
                "jupyter>=1.0.0\n"
                "pyyaml>=6.0\n")
    
    if not os.path.exists("setup.py"):
        create_file("setup.py",
                """
from setuptools import setup, find_packages

setup(
    name="invoice_processing",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.34.0",
        "accelerate>=0.21.0",
        "sentencepiece>=0.1.99",
        "tqdm>=4.66.1",
        "pillow>=10.0.0",
        "matplotlib>=3.7.2",
        "pandas>=2.0.3",
        "python-Levenshtein>=0.21.1",
        "bitsandbytes>=0.41.1",
        "numpy>=1.24.3",
        "pyyaml>=6.0",
    ],
)
""")
    
    if not os.path.exists("README.md"):
        create_file("README.md",
                """# Invoice Processing Project

## Overview
This project focuses on extracting information from invoice images using computer vision and language models.

## Project Structure
- `configs/`: Configuration files for experiments, models, and environments
- `data/`: Dataset storage including images and ground truth
- `models/`: Model storage and cache
- `notebooks/`: Jupyter notebooks for experimentation and analysis
- `results/`: Results storage and visualization
- `src/`: Source code modules
- `scripts/`: Utility scripts
- `docs/`: Documentation

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Set up your environment according to `docs/environment_setup.md`

## Usage
See the notebooks directory for step-by-step guides on running experiments.
""")
    
    if not os.path.exists(".gitignore"):
        create_file(".gitignore",
                """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
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
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# Model files (large binaries)
models/cache/

# Results
results/raw/

# Environment
.env
.venv
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
""")
    
    print(f"\nProject structure created successfully in the current directory: {os.path.abspath('.')}")

if __name__ == "__main__":
    setup_project_in_current_dir()