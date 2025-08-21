# General Agents and World Models

A research project exploring how AI models make decisions in gambling scenarios and betting problems.

## Overview

This project investigates the rationality and decision-making capabilities of language models when presented with probabilistic choices and expected value calculations.

## Installation

```bash
# Install dependencies
uv sync
```

## Usage

### Basic Experiment

Run the main experiment:

```bash
uv run python bets.py
```

### Available Experiments

The `bets.py` script supports three types of experiments:

#### 1. Simple Experiment (`--experiment simple`)

- **Description**: A basic betting scenario using a 10-sided die
- **Bet A**: Win $50 if the die shows 1 (10% probability)
- **Bet B**: Win variable amount if the die shows 2-10 (90% probability)
- **Purpose**: Tests basic expected value calculations in a simple probability scenario

#### 2. Prime Experiment (`--experiment prime`)

- **Description**: A more complex scenario involving prime number identification
- **Bet A**: Win $50 if the die lands on a prime number
- **Bet B**: Win variable amount if the die shows a non-prime number (including 1)
- **Purpose**: Tests the model's ability to reason about mathematical concepts (prime numbers) while making probabilistic decisions
- **Configurable**: Use `--prime_bound` to set the number of sides on the die (default: 1024)

#### 3. Prime Incremental Experiment (`--experiment prime-inc`)

- **Description**: Same as prime experiment but with incremental chain-of-thought reasoning
- **Features**:
  - Gradually reveals reasoning steps to the model
  - Tracks how the model's decision changes as more reasoning is provided
  - Generates plots and tables showing convergence to the optimal decision
- **Purpose**: Studies how step-by-step reasoning affects decision-making quality

### Supported Models

The script supports various language models through OpenRouter:

- **Claude models**: `claude4`, `claude37`
- **GPT models**: `gpt4o`, `gpt41`, `gpt20b`
- **Mistral**: `mistral`
- **Horizon models**: `hor-a`, `hor-b`
- **Kimi**: `kimi-k2`
- **Liquid models**: `lfm-7b`, `lfm-3b`

### Command Line Options

```bash
uv run python bets.py [OPTIONS]

Options:
  --mlflow_exp TEXT        Name of the MLflow experiment for tracking (default: "test")
  --model_name TEXT        Model to use (default: "gpt4o")
  --num_iterations INT     Binary search iterations (default: 10)
  --experiment TEXT        Experiment type: simple/prime/prime-inc (default: "prime")
  --thinking TEXT          Thinking parameter (default: "False")
  --prime_bound INT        Prime bound for prime experiments (default: 1024)
```

### Example Commands

```bash
# Run simple experiment with GPT-4o
uv run python bets.py --experiment simple --model_name gpt4o --mlflow_exp simple_test

# Run prime experiment with Claude
uv run python bets.py --experiment prime --model_name claude4 --mlflow_exp prime_test

# Run incremental experiment with custom prime bound
uv run python bets.py --experiment prime-inc --model_name gpt4o --prime_bound 100 --mlflow_exp incremental_test
```

### Output and Results

- **Standard experiments**: Output the tipping point payout and extracted probabilities
- **Incremental experiments**: Generate plots and CSV tables showing convergence
- **MLflow tracking**: All experiments are logged with parameters and metrics
- **Visualizations**: Incremental experiments create plots showing how decisions evolve with reasoning

## Environment Setup

Create a `.env` file with your API credentials:

```bash
OPENROUTER_API_KEY=your_api_key_here
```

## Testing

```bash
uv run pytest tests/ -v
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
