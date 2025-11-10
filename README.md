# Athena Autocall Pricer  
Monte Carlo Simulation | Local Volatility | CIR Rates | Interactive Streamlit Application

This project provides a complete pricing engine and interactive web application for an Autocall Athena structured product.  
It combines advanced quantitative modeling with a clean and intuitive Streamlit interface, designed for both academic and professional use.

---

## Overview

The application prices an Autocall Athena product through Monte Carlo simulation, using a stochastic interest-rate model and a local-volatility process for the underlying asset.

It allows users to simulate returns, visualize trajectories, test product configurations, and perform historical backtests based on real financial data.

---

## Autocall Athena Logic

The Athena autocall is a structured note whose payoff depends on the performance of an underlying asset, typically an index or a stock.

It features regular observation dates (often annual). At each date:

- Automatic call if the underlying is greater than or equal to its initial level  
  → redemption of the notional plus the coupon  
- Otherwise, the product continues

At maturity:

- If the underlying is above the initial level: redemption of capital plus coupon  
- If it is below the initial level but above the protection barrier: capital is returned  
- If it is below the barrier: capital loss proportional to the underlying's decline  

---

## Mathematical Modeling

The pricing engine relies on two stochastic components:

### Interest Rate Model (CIR)
Interest rates follow the Cox–Ingersoll–Ross (CIR) process:

\[
dr_t = \kappa(\theta - r_t) dt + \sigma_r \sqrt{r_t} dW_t
\]

### Underlying Price Dynamics (Local Volatility)
The underlying price follows a diffusion process with time and level–dependent volatility:

\[
\sigma(t, S_t) = h \left( 2 + \alpha \cos\left( \frac{4\pi t}{T} \right) + \frac{\beta_\sigma t}{1 + S_t^2} \right)
\]

This allows the model to generate realistic market smiles and skews.

---

## Application Usage

The Streamlit interface allows users to input their own assumptions.  
All simulations and plots update automatically.

### Customizable Parameters

- Product maturity (1 to 10 years)  
- Initial interest rate (CIR)  
- Local volatility parameters (h, alpha, beta_sigma)  
- Initial underlying level  
- Annual coupon  
- Protection barrier  
- Number of Monte Carlo paths  

---

## Simulation Results

After parameters are defined, the application calculates:

- Estimated fair value of the Autocall  
- Expected payoff from simulated paths  
- Early redemption probability  
- Loss probability  
- Distribution of terminal payoffs  

---

## Visualizations

The app displays several key plots:

- Simulated interest-rate trajectories  
- Simulated underlying trajectories  
- Discounted underlying paths  
- Payoff distribution histogram  
- Final price distribution  
- Autocall event statistics  

These charts help interpret how different parameters affect early call likelihood and capital risk.

---

## Backtesting Module

A backtest module retrieves historical price data via the `yfinance` API.  
Given a ticker and a time period, the tool:

- Reconstructs historical underlying performance  
- Applies the Autocall payoff rules  
- Detects early call dates  
- Computes the actual payoff that would have been obtained  

This helps evaluate real-world behavior of the product.

---

## Conclusion

This project aims to make structured-product pricing accessible, interactive, and educational.  
It combines quantitative rigor, financial logic, and an intuitive user experience.

It is designed for:

- Students in finance  
- Quantitative analysts  
- Structurers  
- Developers interested in financial engineering  

---

## Author

Mohamed Boumezou  
Université Paris Dauphine – PSL  

