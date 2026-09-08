# Athena Autocall Pricer

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20Pricer-FF4B4B?logo=streamlit&logoColor=white)
![Methods](https://img.shields.io/badge/Methods-Monte%20Carlo%20%7C%20CIR%20%7C%20Local%20Volatility-315C7D)

An interactive Monte Carlo pricer for Athena autocallable structured products, combining local-volatility equity dynamics, stochastic interest rates and historical backtesting.

## Product logic

The model supports the main components of an Athena payoff:

- annual observation dates;
- automatic early redemption when the underlying is above the initial level;
- coupon payment on early call;
- capital protection barrier at maturity;
- downside participation when the final level is below the barrier.

## Application features

- Configurable maturity, coupon, barrier, spot, volatility and simulation count.
- CIR process for the short rate.
- Level- and time-dependent local volatility for the risky asset.
- Monte Carlo path, payoff and redemption-time diagnostics.
- Sensitivity analysis across key product parameters.
- Historical backtesting with market data retrieved through `yfinance`.
- Separate pricing, simulation, sensitivity and backtest views.

## Run locally

```bash
python -m venv .venv
pip install -r requirements.txt
streamlit run app_pricer.py
```

## Repository structure

```text
.
├── app_pricer.py                  # Interactive Streamlit application
├── project.py                     # Standalone modelling workflow
├── ATHENA PRICER - PRESENTATION.pdf
├── Notice - Pricer.pdf            # User guide
├── requirements.txt
└── .devcontainer/
```

## Documentation

- [Project presentation](ATHENA%20PRICER%20-%20PRESENTATION.pdf)
- [Application notice](Notice%20-%20Pricer.pdf)

## Scope and limitations

This project is a transparent educational implementation. It is not a production valuation library and does not include a full calibration framework, issuer credit risk, funding adjustments, transaction costs or all contractual conventions used in live structured-product desks.

## Author

Mohamed Boumezou — Finance student at Université Paris Dauphine–PSL  
[LinkedIn](https://www.linkedin.com/in/mohamed-boumezou-a8a0052ab/)
