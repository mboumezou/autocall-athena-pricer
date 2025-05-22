import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% Fonctions utiles

def sigma(t, x, h, alpha, beta_sigma, T):
    return h * (2 + alpha * np.cos(4 * np.pi * t / T) + (beta_sigma * t) / (1 + x**2))

def compute_payoff_autocall(tildeS, S0, S0_val, z, T, observation_indices, lambda_barrier):
    for idx in observation_indices:
        if tildeS[idx] >= S0_val:
            t_years = idx / 252
            return (S0_val + z * t_years * S0_val) / S0[idx]
        
    if tildeS[-1] >= S0_val:
        return (S0_val + z * T * S0_val) / S0[-1]
    elif tildeS[-1] >= lambda_barrier * S0_val:
        return S0_val / S0[-1]
    else:
        return tildeS[-1] / S0[-1]

def simulate_paths(T, N, r0, h, S0_val):
    # Paramètres internes fixés
    alpha = 0.55
    beta_sigma = 0.42
    a = 0.28
    b = 0.08
    delta = 0.17
    beta_corr = 0.68

    n = 252 * T
    dt = T / n
    t_grid = np.linspace(0, T, n+1)

    r = np.zeros((n+1, N))
    r[0, :] = r0
    S0 = np.ones((n+1, N))
    tildeS = np.ones((n+1, N)) * S0_val
    C = np.zeros((n+1, N))

    for j in range(N):
        for t in range(1, n+1):
            G = np.random.randn()
            Z = np.random.randn()

            r[t, j] = r[t-1, j] + a * (b - r[t-1, j]) * dt + delta * np.sqrt(abs(r[t-1, j])) * np.sqrt(dt) * G
            S0[t, j] = S0[t-1, j] * (1 + r[t-1, j] * dt)
            dC = np.sqrt(dt) * (beta_corr * G + np.sqrt(1 - beta_corr**2) * Z)
            C[t, j] = C[t-1, j] + dC
            vol = sigma(t_grid[t-1], tildeS[t-1, j], h, alpha, beta_sigma, T)
            tildeS[t, j] = tildeS[t-1, j] + vol * tildeS[t-1, j] * dC

    S = tildeS * S0
    return t_grid, r, S0, tildeS, S

def run_backtest(stock_symbol, start_date, end_date, T, z, lambda_barrier, S0):
    import pandas as pd
    import yfinance as yf

    data = yf.download(stock_symbol, start=start_date, end=end_date)
    prices = data['Close']

    initial_date = pd.Timestamp(prices.index[0])
    observation_dates = [initial_date + pd.DateOffset(years=year) for year in range(0, T)]
    initial_price = prices.iloc[0].item()

    payoff_bs = None

    for year, obs_date in enumerate(observation_dates[:-1], 1):
        obs_price = prices.asof(obs_date).item()

        if obs_price >= initial_price:
            payoff_bs = (initial_price * (1 + z * year)) / np.mean(S0[252*year])
            return payoff_bs, year, initial_price

    final_obs_date = observation_dates[-1] if observation_dates[-1] <= prices.index[-1] else prices.index[-1]
    final_price = prices.asof(final_obs_date)

    if final_price >= initial_price:
        payoff_bs = (initial_price * (1 + z * T)) / np.mean(S0[252*T])
    elif final_price >= lambda_barrier * initial_price:
        payoff_bs = initial_price / np.mean(S0[252*T])
    else:
        payoff_bs = final_price / np.mean(S0[252*T])

    return payoff_bs, None, initial_price


# %% Application Streamlit

st.title("Athena - Autocall Pricer 🚀")

with st.expander("📖 Présentation du produit Autocall Athena"):
    st.write("""
    L'Autocall Athena est un produit structuré sur actions.

    **Fonctionnement :**
    - Chaque année, si le sous-jacent est au-dessus de son niveau initial, remboursement automatique avec capital + coupon.
    - Sinon, le produit continue.
    - À maturité :
        - Si sous-jacent > niveau initial ➔ capital + coupon.
        - Si sous-jacent > barrière ➔ remboursement du capital.
        - Sinon ➔ perte proportionnelle.

    **Avantage :** possibilité de rappels rapides.  
    **Risque :** perte du capital si forte baisse du sous-jacent.
    """)


st.sidebar.header("Paramètres de simulation")

T = st.sidebar.slider("Nombre d'années de simulation", 1, 10, 4)

r0 = st.sidebar.number_input(
    "Taux sans risque initial", value=0.01,
    step=0.0001,
    format="%.4f"
)

h = st.sidebar.number_input("Niveau de volatilité locale (h)", value=0.08)
S0_val = st.sidebar.number_input("Niveau initial du sous-jacent", value=100.0)

st.sidebar.info("Note :\n- h ≈ 0.08 pour un stock stable (CAC 40)\n- h ≈ 0.12 pour un stock plus volatil\n- h ≈ 0.16+ pour un stock très risqué (tech, biotech)")

z = st.sidebar.number_input(
    "Taux de coupon annuel (en décimal, ex: 0.05 pour 5%)",
    value=0.05,
    step=0.0001,
    format="%.4f"
)
lambda_barrier = st.sidebar.slider("Barrière de protection (en % de S0)", 0.5, 1.0, 0.7)
N = 300 

t_grid, r, S0, tildeS, S = simulate_paths(T, N, r0, h, S0_val)

observation_indices = [252*k for k in range(1, T)]

payoffs = []

for j in range(N):
    payoff = compute_payoff_autocall(tildeS[:, j], S0[:, j], S0_val, z, T, observation_indices, lambda_barrier)
    payoffs.append(payoff)

price_payoff = np.mean(payoffs)

# Affichage des résultats
st.header("Résultats")
st.metric(label="Prix estimé de l'Autocall Athena", value=f"{price_payoff:.2f}")

# Graphiques des trajectoires
st.subheader("Trajectoires simulées")

fig, axs = plt.subplots(2, 2, figsize=(14, 8))

for j in range(10):
    axs[0, 0].plot(t_grid, r[:, j])
axs[0, 0].set_title("Trajectoires de $r_t$")
axs[0, 0].grid(True)

for j in range(10):
    axs[0, 1].plot(t_grid, S0[:, j])
axs[0, 1].set_title("Trajectoires de $S_t^0$")
axs[0, 1].grid(True)

for j in range(10):
    axs[1, 0].plot(t_grid, tildeS[:, j])
axs[1, 0].set_title("Trajectoires de $\\tilde{S}_t$")
axs[1, 0].grid(True)

for j in range(10):
    axs[1, 1].plot(t_grid, S[:, j])
axs[1, 1].set_title("Trajectoires de $S_t$")
axs[1, 1].grid(True)

plt.tight_layout()
st.pyplot(fig)

# Histogramme des payoffs
st.subheader("Distribution des Payoffs")

fig2 = plt.figure(figsize=(10, 6))
plt.hist(payoffs, bins=50, edgecolor='black', alpha=0.75)
plt.title("Distribution des Payoffs de l'Autocall")
plt.xlabel("Payoff actualisé")
plt.ylabel("Fréquence")
plt.grid(True, linestyle='--', alpha=0.7)
st.pyplot(fig2)






st.header("Backtest historique 📈")

st.subheader("Paramètres du backtest")
stock_symbol = st.text_input("Ticker du sous-jacent (ex: TTE.PA)", value="TTE.PA")
start_date = st.date_input("Date de début", value=pd.to_datetime("2021-01-01"))
end_date = st.date_input("Date de fin", value=pd.to_datetime("2025-01-01"))

if st.button("Lancer le backtest"):
    payoff_bs, year, initial_price = run_backtest(stock_symbol, start_date, end_date, T, z, lambda_barrier, S0)

    st.write(f"**Prix initial du sous-jacent : {initial_price:.2f} €**") 

    if year is not None:
        st.success(f"Produit rappelé après {year} ans ! Payoff observé = {payoff_bs:.2f}")
    else:
        st.info(f"Produit arrivé à maturité. Payoff observé = {payoff_bs:.2f}")


