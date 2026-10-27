# Ollama + MLflow Model Evaluation Framework

A local model-evaluation framework for benchmarking, testing, and comparing open-source large language models using Ollama and MLflow.

The project evaluates model behaviour across baseline, robustness, and regression test rounds. It records evaluation results in JSON reports and uses MLflow to track experiment runs.

## Project Goals

- Compare local LLMs using repeatable evaluation cases.
- Validate expected outputs, including strict JSON responses.
- Test behaviour under ambiguous and uncertainty-sensitive prompts.
- Detect regressions against configured baseline expectations.
- Record evaluation latency and available performance metrics.
- Maintain reproducible reports for technical analysis and model selection.

## Models Evaluated

The configured models are:

| Model | Purpose |
|---|---|
| `llama3.2:3b` | Local LLM benchmark |
| `qwen2.5:3b` | Local LLM benchmark |
| `gemma2:2b` | Local LLM benchmark |

Results depend on the machine, model state, prompt, runtime conditions, and evaluation configuration. Measurements from this project should not be treated as universal rankings of these models.

## Technology Stack

- **Python** — evaluation logic and automation
- **Ollama** — local LLM inference
- **MLflow** — experiment and run tracking
- **pytest** — automated testing
- **JSON** — test configuration and evaluation reports
- **psutil** — system resource monitoring where configured

## Architecture

```text
Test Cases (JSON)
       |
       v
Evaluation Runner
       |
       v
Evaluation Engine
       |
       +------> Ollama API ------> Local LLMs
       |                            |
       |<------ Model Responses <---+
       |
       +------> Baseline Checks
       |
       +------> Robustness Checks
       |
       +------> Regression Checks
       |
       +------> JSON Evaluation Reports
       |
       +------> MLflow Experiment Tracking
```

## Project Structure

```text
evaluate_trial/
├── config/
│   └── test_cases.json
├── reports/
│   ├── baseline_all_models.json
│   ├── robustness_all_models.json
│   └── regression_all_models.json
├── tests/
│   ├── test_configuration.py
│   ├── test_evaluation_engine.py
│   ├── test_mlflow_tracker.py
│   └── test_regression.py
├── evaluate_models.py
├── evaluation_engine.py
├── mlflow_tracker.py
├── run_evaluation.py
├── restore_mlflow.py
├── .gitignore
└── README.md
```

Additional model-specific reports and supporting scripts are also included.

## Prerequisites

- Windows, Linux, or macOS
- Python compatible with the installed project dependencies
- [Ollama](https://ollama.com/)
- [Git](https://git-scm.com/) for version control

The following commands illustrate the initial setup on a Windows PowerShell terminal. Run them from the project directory when appropriate.

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Python dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install mlflow requests psutil pytest
```

If additional dependencies are required by your installed project code, install those dependencies in the same environment.

### 3. Pull the configured Ollama models

Make sure Ollama is installed and running, then execute:

```powershell
ollama pull llama3.2:3b
ollama pull qwen2.5:3b
ollama pull gemma2:2b
```

Confirm the available models:

```powershell
ollama list
```

## Running the Evaluation

From the project root, run each evaluation round separately:

### Baseline

```powershell
python run_evaluation.py baseline
```

Baseline tests check configured expected behaviour and establish reference results.

### Robustness

```powershell
python run_evaluation.py robustness
```

Robustness tests examine responses to ambiguous prompts, strict output-format requirements, and uncertainty-sensitive questions.

### Regression

```powershell
python run_evaluation.py regression
```

Regression tests compare current behaviour against configured baseline expectations and thresholds.

### Automated tests

```powershell
python -m pytest .\tests\ -v
```

The project's test suite previously completed with **20 passed**. Rerun the tests after making code or configuration changes to confirm the current state.

## Viewing Results in MLflow

The project has used a local SQLite tracking database named `benchmark_mlflow.db`.

From the project directory, start the MLflow interface:

```powershell
mlflow ui --backend-store-uri "sqlite:///D:/AI Engineering 2/evaluate_trial/benchmark_mlflow.db" --host 127.0.0.1 --port 5000
```

Keep the terminal running and open:

**http://127.0.0.1:5000**

Select the configured experiment, `Ollama Benchmark Framework`, to inspect recorded runs and available metrics.

If runs are missing, verify that MLflow is using the same tracking database as the evaluation script. Do not delete or replace existing tracking databases to troubleshoot the UI.

## Evaluation Results

The following results were recorded during one local evaluation session.

| Evaluation round | Passed | Failed | Pass rate |
|---|---:|---:|---:|
| Baseline | 9 | 0 | 100% |
| Robustness | 10 | 2 | 83.3% |
| Regression | 9 | 0 | 100% |
| **Total executions** | **28** | **2** | **93.3%** |

The total includes repeated test cases across evaluation rounds; it does not represent 30 unique test cases.

### Baseline average latency

| Model | Average latency |
|---|---:|
| `llama3.2:3b` | 86.77 seconds |
| `qwen2.5:3b` | 104.44 seconds |
| `gemma2:2b` | 60.27 seconds |

These averages cover only three baseline cases per model. They are preliminary measurements, not conclusive performance rankings. Model loading, warm-up, and system load can affect latency.

### Robustness finding

Two robustness checks failed because Qwen and Gemma returned JSON wrapped in Markdown code fences, which the strict JSON parser rejected. The underlying inference requests succeeded, but the responses did not satisfy the configured format validator.

Llama passed all four robustness cases in this evaluation.

### Regression finding

All nine regression executions passed, and the configured regression comparison reported `no_threshold_exceeded` for all three models. This means no configured threshold was exceeded in that comparison; it does not prove that the models are free of defects.

### Evaluation limitations

- A non-empty response does not establish factual correctness.
- Strict-format validation is distinct from inference success.
- Small test suites cannot establish broad model reliability.
- Latency varies with hardware, model loading, and runtime conditions.
- Resource measurements should be interpreted in the context of when and how they were sampled.

## Reports and Reproducibility

Evaluation output is stored in JSON files under `reports/`, with additional summary files in the project root.

The reports preserve the outputs and checks from the recorded evaluation runs. Run the evaluations again to obtain new measurements after changing models, prompts, thresholds, or code.

## Engineering Practices

This project demonstrates practical work with:

- Automated LLM evaluation
- Test-driven validation
- Baseline and regression checking
- Output-format validation
- Local model serving
- Experiment tracking
- Structured evaluation reporting
- Reproducible testing workflows

## Future Improvements

Potential next steps include:

- Add semantic similarity and reference-based answer-quality metrics.
- Expand factuality and hallucination evaluation using verified references.
- Measure token throughput and time to first token consistently.
- Separate cold-start and warm-start latency measurements.
- Add repeated trials and confidence intervals.
- Track hardware and software environment metadata with every run.
- Integrate automated report generation into a CI pipeline.
- Add configurable model-specific quality and latency thresholds.

## License

No license has been specified yet. Add a license before inviting others to reuse, modify, or distribute this project.

## Disclaimer

This is an experimental local evaluation framework. Its results are intended to support engineering analysis and should not be interpreted as a comprehensive certification of model safety, factuality, or production readiness.
