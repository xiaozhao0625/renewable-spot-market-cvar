# Data and Code

This archive contains the data, analysis code, and numerical results for
“Multi-Scenario-Based Optimization with CVaR Risk Control for Renewable Energy Participation in Electricity Spot Markets”.

## Contents

- `code/generate_source_data.py`: constructs the hourly five-scenario source dataset.
- `code/run_analysis.py`: performs scenario generation, fast-forward scenario reduction, bidding optimization, independent out-of-sample evaluation, sensitivity analysis, and figure generation.
- `data/multi_scenario_power_market_dataset.csv`: hourly scenario source dataset for 2025.
- `data/hourly_out_of_sample_summary.csv`: hourly bids and out-of-sample summary statistics.
- `results/out_of_sample_performance.csv`: complete comparison of the three bidding strategies.
- `results/scenario_reduction_audit.csv`: retained scenario indices, probabilities, and reduction distances for each day.
- `results/monthly_out_of_sample_revenue.csv`: monthly revenue totals by strategy.
- `results/risk_aversion_sensitivity.csv`: out-of-sample sensitivity results for the risk-aversion coefficient.
- `results/daily_computation_times.csv`: daily execution times.
- `results/environment_and_runtime.json`: model settings, software versions, and runtime environment.
- `figures/*.svg`: figures generated directly by the analysis script.

## Requirements

- Python 3.12 or later
- NumPy 2.3.5
- pandas 3.0.1

Install the required packages with:

```text
python -m pip install -r requirements.txt
```

## Reproduce the analysis

Run the analysis from the package root:

```text
python code/run_analysis.py
```

The script reads the included source dataset and overwrites the derived data,
result tables, runtime records, and SVG figures with a fresh reproducible run.
The training and test random seeds and all model parameters are defined in the
`Config` dataclass near the beginning of the script.

To regenerate the source dataset before running the analysis:

```text
python code/generate_source_data.py
python code/run_analysis.py
```

## Evaluation design

For each day, 200 training trajectories are generated and reduced to five
representative scenarios using fast-forward selection. The deterministic,
risk-neutral, and CVaR-adjusted bids are then fixed and evaluated on 1,000
independently seeded evaluation trajectories. Evaluation trajectories are not
used for scenario reduction, bid selection, or parameter tuning.

The source series are produced by the documented scenario data-generating
process and do not contain measured market observations. The results therefore
support evaluation under that process and should not be interpreted as
deployment evidence for a specific electricity market.
