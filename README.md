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

Run the main experiment:

```bash
uv run python bets.py
```

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
