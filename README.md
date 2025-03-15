# Invoice Processing Project

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

## Project Configuration and Setup

### Configuration System
This project uses a layered configuration system:
- **Environment Configuration**: Handles environment-specific settings (local vs. RunPod)
- **Path Configuration**: Manages file paths across different environments
- **Experiment Configuration**: Controls experiment parameters, metrics, and visualization

Configuration files are stored in YAML format in the `configs/` directory, making it easy to modify settings without changing code.

### Environment Setup
The project includes scripts for environment setup:
- Windows: `scripts/setup_environment.bat`
- Linux/RunPod: `scripts/setup_environment.sh`

These scripts install dependencies, create necessary directories, and validate your environment.

### Setup Notebook
The `01_environment_setup.ipynb` notebook provides a step-by-step process to set up and verify your working environment:

1. **System Check**: Verifies Python, PyTorch, and GPU availability
2. **Project Structure**: Locates project root and ensures all directories exist
3. **Configuration Loading**: Loads environment-specific settings (local or RunPod)
4. **Experiment Preparation**: Previews different experiment types (prompt comparison, model comparison)
5. **Data Verification**: Checks for ground truth data and invoice images
6. **Dependency Installation**: Automatically installs all required dependencies based on your environment

To use the setup notebook:
1. Open `notebooks/01_environment_setup.ipynb` in Jupyter
2. Run all cells in sequence
3. Review the output to ensure your environment is properly configured
4. The notebook saves setup information to `results/{timestamp}/setup_info.json` for reference

#### RunPod-Specific Setup

When running on RunPod, the setup notebook will:
- Automatically detect the RunPod environment
- Install all required dependencies (no need for manual pip install commands)
- Configure optimal paths for model caching in the `/cache` directory (RunPod's persistent storage)
- Set up environment variables for efficient model loading

This ensures models like Pixtral-12B are stored persistently between sessions, eliminating the need to re-download large models each time you start a new RunPod instance.

Run this notebook before starting any experiments to ensure your environment is correctly configured. This notebook should be run again whenever you change your working environment (e.g., moving from local to RunPod).

## Model Loading Framework

### Overview
The model loading framework provides a structured approach to loading and configuring large vision-language models like Pixtral-12B. It separates model configuration from loading logic, making it easy to experiment with different models and optimization strategies.

### Components

#### Model Configuration System
- **YAML Configuration Files**: Located in `configs/models/` directory
- **Structured Configuration Classes**: Defined in `src/models/model_configs.py`
- **Validation Logic**: Ensures configuration accuracy before model loading

#### Model Registry
The registry (`src/models/registry.py`) serves as a catalog of available models and provides:
- Model discovery and listing capabilities
- Configuration loading and caching
- Environment-specific overrides
- Access to loading parameters with different quantization options

#### Model Loader
The loader (`src/models/loader.py`) handles the actual model instantiation:
- Loads models with appropriate configuration for different environments
- Applies memory optimization techniques (precision, quantization)
- Monitors and reports GPU memory usage
- Validates model loading and handles errors

### Usage

#### Basic Model Loading
```python
from src.models.loader import load_model_and_processor

# Load model with default configuration
model, processor = load_model_and_processor("pixtral-12b")

# Load with specific quantization strategy
model, processor = load_model_and_processor("pixtral-12b", quantization="int8")
```

#### GPU Compatibility Checking
```python
from src.models.loader import verify_gpu_compatibility

# Check if current GPU meets model requirements
compatibility = verify_gpu_compatibility("pixtral-12b")
if compatibility["compatible"]:
    print("GPU is compatible with model requirements")
else:
    print(f"GPU is not compatible: {compatibility['reason']}")
```

#### Memory Management
```python
from src.models.loader import get_gpu_memory_info, optimize_memory

# Get current GPU memory information
memory_info = get_gpu_memory_info()
print(f"GPU memory usage: {memory_info['allocated_memory_gb']:.2f} GB")

# Optimize memory after model use
optimize_memory(clear_cache=True)
```

### Model Configuration Format

Model configuration files in `configs/models/` follow this structure:

```yaml
# Basic model information
name: "model-name"
repo_id: "repository-id"
model_type: "ModelClass"
processor_type: "ProcessorClass"
description: "Model description"

# Hardware requirements
hardware:
  gpu_required: true
  gpu_memory_min: "24GB"
  recommended_gpu: "A4000 or better"

# Loading configuration
loading:
  default_strategy: "optimized"
  device_map: "cuda:0"

# Quantization options
quantization:
  default: "bfloat16"
  options:
    bfloat16:
      torch_dtype: "bfloat16"
      device_map: "cuda:0"
    int8:
      load_in_8bit: true
      device_map: "auto"
    int4:
      load_in_4bit: true
      bnb_4bit_compute_dtype: "bfloat16"
      device_map: "auto"

# Inference parameters
inference:
  max_new_tokens: 50
  do_sample: false
  batch_size: 1
  temperature: 1.0
```

### Adding New Models

To add a new model to the framework:

1. Create a new YAML file in `configs/models/` directory
2. Define the model's configuration following the format above
3. The model will automatically be available through the registry

### Quantization Strategies

The framework supports multiple quantization strategies:

- **bfloat16**: Uses bfloat16 precision (default, balance of speed and accuracy)
- **int8**: 8-bit quantization for reduced memory usage
- **int4**: 4-bit quantization for maximum memory efficiency

Different strategies can be selected at runtime without changing configuration files.

## Prompt Management System

### Overview
The Prompt Management System provides a structured framework for organizing, accessing, and experimenting with different prompts for invoice information extraction. It enables systematic testing of various prompt formulations to identify the most effective approaches for each field type.

### Components

#### Prompt Registry
The registry (`src/prompts/registry.py`) serves as the central storage for all prompts:

- Stores prompts with metadata including category, field, and version
- Provides retrieval by name, category, or field type
- Handles loading from and saving to configuration files
- Supports iteration through filtered prompt subsets

#### Invoice-Specific Prompts
Domain-specific prompts (`src/prompts/invoice_prompts.py`) implement various strategies for invoice data extraction:

- Organizes prompts by extraction field (work order, cost, date)
- Provides different prompt categories (basic, detailed, positioned)
- Includes utility functions for accessing recommended prompts
- Automatically initializes the registry with default prompts

#### Prompt Utilities
Utility functions (`src/prompts/prompt_utils.py`) for working with prompts:

- Model-specific formatting for different vision-language models
- Templating and variable substitution for prompt creation
- Analysis tools to calculate prompt complexity metrics
- Batch processing capabilities for efficient experimentation

#### Configuration Files
YAML configuration files in `configs/prompts/` store structured prompt definitions:

- `basic.yaml`: Simple, direct prompts with minimal instructions
- `detailed.yaml`: Elaborate prompts with field descriptions and context
- `positioned.yaml`: Prompts with spatial information about field locations

### Integration with Experiments
The Prompt Management System integrates with the experiment configuration system:

- `ExperimentConfig.get_prompts_for_experiment()`: Retrieves prompts based on experiment parameters
- `ExperimentConfig.validate_prompt_config()`: Validates prompt-related configuration
- `ExperimentConfig.format_prompt_for_model()`: Prepares prompts for specific models
- `ExperimentConfig.create_prompt_experiment()`: Creates configurations for prompt comparison

### Usage

#### Basic Prompt Access
```python
from src.prompts import get_prompt, format_prompt

# Get a specific prompt by name
prompt = get_prompt("basic_work_order")

# Format it for a specific model
formatted_text = format_prompt(prompt, "pixtral-12b")
# Result: "<s>[INST]Extract the work order number from this invoice image.\n[IMG][/INST]"
```

#### Getting Prompts for Experiments
```python
from src.config.experiment import get_experiment_config

# Get experiment configuration
config = get_experiment_config("prompt_comparison")

# Get prompts for this experiment
experiment_prompts = config.get_prompts_for_experiment()

# Format each prompt for the model
for prompt_info in experiment_prompts:
    formatted = config.format_prompt_for_model(prompt_info["name"])
    # Use formatted prompt for inference...
```

#### Creating Prompt Variations
```python
from src.prompts import create_prompt_variants

# Create systematic variations of a prompt
base_prompt = "Extract the {field} from the {location} of this invoice."
variations = {
    "field": ["work order number", "order ID", "job number"],
    "location": ["top section", "header", "upper right corner"]
}

variants = create_prompt_variants(base_prompt, variations)
# Generates 9 different prompt combinations
```

### Adding New Prompts

New prompts can be added in three ways:

1. Via Configuration Files

Add entries to YAML files in `configs/prompts/` following this format:
```yaml
prompts:
  - name: "new_prompt_name"
    text: "Your prompt text here."
    category: "your_category"
    field_to_extract: "work_order"
    description: "Description of what this prompt tests."
    version: "1.0"
    metadata:
      source: "your_identifier"
```

2. Via Source Code

Add prompts programmatically in `src/prompts/invoice_prompts.py`:
```python
register_prompt(Prompt(
    name="new_work_order_prompt",
    text="Your prompt text here.",
    category="your_category",
    field_to_extract: "work_order",
    description="Description of what this prompt tests."
))
```

3. From Notebooks

Create and register prompts directly from experiment notebooks:
```python
from src.prompts import Prompt, register_prompt

register_prompt(Prompt(
    name="experiment_prompt",
    text="Your prompt text here.",
    category="experimental",
    field_to_extract="work_order",
    description="Created during experimentation."
))
```

### Prompt Categories
The system organizes prompts into several categories:

- **basic**: Simple, direct instructions with minimal context
- **detailed**: Elaborate prompts with field descriptions and multiple label options
- **positioned**: Prompts that direct attention to specific areas of the invoice
- **formatted**: Prompts that specify output format requirements
- **contextual**: Prompts that provide business context around the document
- **chain_of_thought**: Prompts that guide the model through step-by-step reasoning

This categorization enables systematic testing to determine which prompt strategies work best for different extraction tasks.

## Extraction Pipeline Framework

### Overview
The Extraction Pipeline Framework provides a comprehensive system for orchestrating the entire extraction workflow, from loading models and prompts to processing images, analyzing results, and generating visualizations. This pipeline architecture enables systematic experimentation with different models, prompts, and extraction strategies.

### Components

#### Pipeline Orchestration
The central orchestration class (`src/execution/pipeline.py`) coordinates the entire workflow:
- Manages experiment configuration and initialization
- Handles model and processor loading
- Coordinates batch processing of images
- Collects and analyzes results
- Organizes outputs into appropriate directories

#### Inference Engine
The inference engine (`src/execution/inference.py`) provides core image processing capabilities:
- Processes individual images using vision-language models
- Extracts field information based on prompts
- Calculates accuracy metrics against ground truth
- Monitors performance and resource usage

#### Batch Processing
The batch processing system (`src/execution/batch.py`) efficiently handles multiple images:
- Creates optimal batch sizes based on GPU memory constraints
- Implements checkpointing for resumable processing
- Provides progress tracking and error handling
- Optimizes memory usage between batches

#### Result Organization
Results are automatically organized into a structured directory hierarchy:
- **Raw Results**: Detailed extraction outputs for each image, stored in `experiment_dir/raw/`
- **Processed Results**: Summary metrics and analysis, stored in `experiment_dir/processed/`
- **Visualizations**: Charts, graphs, and dashboards, stored in `experiment_dir/visualizations/`

### Pipeline Configuration

The pipeline's behavior is controlled through configuration files in `configs/pipeline/`:

```yaml
# General experiment settings
experiment:
  description: "Default pipeline configuration for invoice field extraction"
  model_name: "pixtral-12b"
  field_to_extract: "work_order"
  prompt_category: "specific"

# Batch processing settings
batch_processing:
  auto_batch_size: true
  max_batch_size: 8
  memory_threshold: 0.8
  optimize_between_batches: true

# Checkpointing configuration
checkpointing:
  enable: true
  frequency: 5
  resume: true

# Error handling and resources
error_handling:
  continue_on_error: true
  max_failures: 50
resources:
  cleanup_on_complete: true
  monitor_memory: true

# Output settings
output:
  show_progress: true
  metrics: ["exact_match", "character_error_rate"]
```

### Usage

#### Basic Extraction
```python
from src.execution.pipeline import run_extraction_pipeline

# Run complete pipeline with single function call
pipeline = run_extraction_pipeline(
    experiment_name="work_order_extraction_test", 
    model_name="pixtral-12b",
    field_type="work_order",
    prompt_name="basic_work_order"
)

# Access results and summary
results = pipeline.results
summary = pipeline.analyze_results()
```

#### Customized Pipeline
```python
from src.execution.pipeline import ExtractionPipeline

# Create pipeline instance
pipeline = ExtractionPipeline(experiment_name="custom_extraction")

# Load ground truth data
pipeline.load_ground_truth()

# Setup model
pipeline.setup_model(model_name="pixtral-12b", quantization="int8")

# Get prompt
prompt = pipeline.get_experiment_prompt(prompt_name="detailed_work_order")

# Run extraction
results = pipeline.run_extraction(
    field_type="work_order",
    prompt=prompt,
    batch_size=4
)

# Analyze and save results
summary = pipeline.analyze_results()
pipeline.save_results(summary=summary)

# Clean up resources
pipeline.cleanup()
```

### Comparative Experiments

#### Prompt Comparison
```python
# Compare different prompts for the same field
comparison = pipeline.run_prompt_comparison(
    field_type="work_order",
    prompt_category="all",
    num_images=20
)

# Results are saved to the processed directory
# and returned for immediate analysis
```

#### Model Comparison
```python
# Compare different models with the same prompt
model_comparison = pipeline.run_model_comparison(
    model_names=["pixtral-12b", "llava-7b", "another-model"],
    field_type="work_order",
    prompt_name="basic_work_order",
    num_images=20
)
```

### Experiment Workflow

The pipeline supports three primary experiment types:

1. **Single Model, Single Prompt**: Basic proof of concept and baseline establishment
2. **Single Model, Multiple Prompts**: Systematic testing of different prompt strategies
3. **Multiple Models, Multiple Prompts**: Comprehensive comparison across models and prompts

Each experiment follows the same general workflow:
1. **Setup**: Initialize the pipeline and load necessary components
2. **Execution**: Process images in batches with appropriate checkpoint handling
3. **Analysis**: Calculate metrics and generate insights from results
4. **Visualization**: Create visual representations of performance
5. **Cleanup**: Release resources and prepare for subsequent experiments

### Result Analysis and Visualization

The pipeline automatically generates analysis and visualizations:

- **Accuracy Metrics**: Exact match rate, character error rate
- **Performance Metrics**: Processing time, GPU memory usage
- **Visual Reports**: Charts showing accuracy distribution, error patterns
- **Failure Analysis**: Detailed breakdown of extraction issues
- **HTML Dashboard**: Interactive summary of experiment results

All visualizations are stored in the experiment's visualization directory for easy reference and comparison.