# QuantMatrix

A modular, event-driven quantitative trading system with a microservices architecture.

## Architecture

- **Data Module**: Fetches, processes, and stores market data
- **Analysis Module**: Technical indicators, statistical analysis, and visualization
- **Strategy Module**: Strategy framework, backtesting, and optimization
- **Execution Module**: Order management, risk control, and broker interfaces
- **Monitoring System**: Dashboard, alerts, and logging

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Configure data sources in `config/data_sources.json`
3. Run the system: `python main.py`

## Development

- Python 3.8+
- Uses Docker for containerization
- CI/CD with GitHub Actions 