# Athena Autocall Pricer

This project implements a Monte Carlo pricer for an Autocall Athena structured product, combined with an interactive Streamlit interface. It is designed to provide a clear, flexible and accessible framework for pricing, analysing, and testing Autocall payoff structures.

## Features

- Monte Carlo simulation of the underlying asset using a local volatility model.
- Stochastic interest rates based on the CIR process.
- Full Autocall Athena payoff logic:
  - Annual observation dates.
  - Automatic early redemption if the underlying is above the initial level.
  - Capital protection barrier at maturity.
  - Capital loss below the barrier.
- Adjustable parameters: maturity, coupon, barrier, initial level, volatility parameters, interest-rate level, and number of simulations.
- Real-time visualisation of simulated paths and payoff distributions.
- Historical backtesting using market data retrieved with `yfinance`.

## Overview of the Model

### Underlying dynamics
The underlying price evolves under a local volatility model, allowing time-dependent and level-dependent volatility.

### Interest rates
Rates follow the Cox–Ingersoll–Ross (CIR) stochastic process.

### Pricing method
The Autocall payoff is evaluated over thousands of simulated paths and discounted to obtain its fair value.

## Backtesting

The application includes a backtest module that simulates how the product would have behaved using historical market data. It identifies early call dates and computes realised payoffs over the selected period.

## Purpose

This project aims to provide a practical tool for students, quants, and structurers who want to understand or experiment with Autocall pricing in a transparent and configurable environment.

## Author

Mohamed Boumezou  
Université Paris Dauphine – PSL
