# %% Paramètres initiaux
import numpy as np
import matplotlib.pyplot as plt

while True:
    try:
        T=int(input("Entrer le nombre d'années de simulation :"))
        if T > 0:
            break
        else:
            print("Merci d'enter un entier strictement positif")
    except ValueError:
        print("Entrée invalide. Merci d'entrer un nombre entier")

n = 252 * T
dt = T/n
N = 1000

t_grid = np.linspace(0, T, n+1)
observation_indices = [252*k for k in range(1,T)]

print(observation_indices)

while True:
    try:
        r0 = float(input("Entrez le taux sans risque actuel : "))
        break
    except ValueError:
        print("Entrez un nombre correct")

b = 0.08
a = 0.28
delta = 0.17
beta_corr = 0.68

while True:
    try:
        h = float(input("Entrez le niveau de volatilité locale h (ex: 0.08 pour un stock classique) : "))
        if h > 0 and h<1:
            break
        else:
            print("Merci d'entrer une valeur positive.")
    except ValueError:
        print("Entrée invalide, veuillez entrer un nombre.")

alpha = 0.55
beta_sigma = 0.42

while True:
    try:
        S0_val = float(input("Entrez le niveau initial du sous-jacent :"))
        if S0_val > 0 :
            break
    except ValueError:
        print("Le niveau initial saisi est incorrect")

while True:
    try:
        z = float(input("Entrez le taux de coupon :"))
        if z > 0 and z<=1 :
            break
    except ValueError:
        print("Le taux de coupon saisi est incorrect")

while True:
    try:
        lambda_barrier = float(input("Entrez la barrière capital du sous-jacent :"))
        if lambda_barrier > 0 and lambda_barrier <= 1 :
            break
    except ValueError:
        print("Le niveau  saisi est incorrect")



# %% Initialisations
r = np.zeros((n+1,N))
r[0, :] = r0

print(r)

S0 = np.ones((n+1,N))
tildeS = np.ones((n+1,N)) * S0_val
C = np.zeros((n+1,N))

def sigma(t, x):
    return h * (2 + alpha * np.cos(4 * np.pi * t / T) + (beta_sigma * t) / (1 + x**2))


# %% Simulations

for j in range(N):
    for i in range(1, n+1):
        G = np.random.randn()
        Z = np.random.randn()
        
        # Simulation de r_t
        r[i, j] = r[i-1, j] + a * (b - r[i-1, j]) * dt + delta * np.sqrt(abs(r[i-1, j])) * np.sqrt(dt) * G
        
        # Simulation de l'actif sans risque S0_t
        S0[i, j] = S0[i-1, j] * (1 + r[i-1, j] * dt)
        
        # Simulation du mouvement brownien corrélé C_t
        dC = np.sqrt(dt) * (beta_corr * G + np.sqrt(1 - beta_corr**2) * Z)
        C[i, j] = C[i-1, j] + dC
        
        # Simulation de tildeS_t (actif risqué actualisé)
        vol = sigma(t_grid[i-1], tildeS[i-1, j])
        tildeS[i, j] = tildeS[i-1, j] + vol * tildeS[i-1, j] * dC

S = tildeS * S0

# %% Fonction payoff 

def compute_payoff_autocall(tildeS, S0, S0_val, i, T, observation_indices, lambda_barrier):

    for idx in observation_indices:
        if tildeS[idx] >= S0_val:
            t_years = idx/252
            return (S0_val + z*t_years*S0_val) / S0[idx]
        
    if tildeS[-1] >= S0_val:
        return (S0_val + z*T) / S0[-1]
    elif tildeS[-1] >= lambda_barrier * S0_val:
        return S0_val / S0[-1]
    else:
        return tildeS[-1] / S0[-1]
    
# %% Calcul des payoff

payoffs=[]

for j in range (N):
    payoff = compute_payoff_autocall(tildeS[:, j], S0[:, j], S0_val, z, T, observation_indices, lambda_barrier)
    print(payoff)
    payoffs.append(payoff)


# %% Moyenne Monte Carlo
price_payoff = np.mean(payoffs)
print(price_payoff)


# %% Affichage des trajectoires simulées

fig, axs = plt.subplots(2, 2, figsize=(14, 8))

# Trajectoires du taux sans risque r_t
for j in range(10):
    axs[0, 0].plot(t_grid, r[:, j])
axs[0, 0].set_title("Trajectoires simulées de $r_t$")
axs[0, 0].set_xlabel("Temps")
axs[0, 0].set_ylabel("$r_t$")
axs[0, 0].grid(True)

# Trajectoires de l'actif sans risque S_t^0
for j in range(10):
    axs[0, 1].plot(t_grid, S0[:, j])
axs[0, 1].set_title("Trajectoires simulées de $S_t^0$")
axs[0, 1].set_xlabel("Temps")
axs[0, 1].set_ylabel("$S_t^0$")
axs[0, 1].grid(True)

# Trajectoires de l'actif actualisé tilde{S}_t
for j in range(10):
    axs[1, 0].plot(t_grid, tildeS[:, j])
axs[1, 0].set_title("Trajectoires simulées de $\\tilde{S}_t$")
axs[1, 0].set_xlabel("Temps")
axs[1, 0].set_ylabel("$\\tilde{S}_t$")
axs[1, 0].grid(True)

# Trajectoires de l'actif réel S_t
for j in range(10):
    axs[1, 1].plot(t_grid, S[:, j])
axs[1, 1].set_title("Trajectoires simulées de $S_t$")
axs[1, 1].set_xlabel("Temps")
axs[1, 1].set_ylabel("$S_t$")
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()


# %%
plt.figure(figsize=(10,6))
plt.hist(payoffs, bins=50, edgecolor='black', alpha=0.75)
plt.title("Distribution des Payoffs de l'Autocall Athena", fontsize=16)
plt.xlabel("Payoff actualisé", fontsize=14)
plt.ylabel("Fréquence", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# %% Backtest

import pandas as pd
import yfinance as yf

data = yf.download("TTE.PA", start="2021-01-01", end="2025-01-01")
prices = data['Close']

payoffs_bs = None

initial_date = pd.Timestamp(prices.index[0])
observation_dates = [initial_date + pd.DateOffset(years=year) for year in range(0, T)]
initial_price = prices.iloc[0].item()

for year, obs_date in enumerate(observation_dates[:-1], 1):

    obs_price = prices.asof(obs_date).item()


    
    if obs_price >= initial_price:
        payoff_bs = (initial_price * (1+z*year)) / np.mean(S0[252*year])
        print(f"Produit rappelé à {year} ans, payoff = {payoff_bs}")
        break

if payoff_bs is None:
    final_obs_date = observation_dates[-1] if observation_dates[-1] <= prices.index[-1] else prices.index[-1]
    final_price = prices.asof(final_obs_date)

    if final_price >= initial_price:
        payoff_bs= initial_price * (1 + (z * T)) / np.mean(S0[252*T])
    elif final_price >= lambda_barrier * initial_price:
        payoff_bs = initial_price / np.mean(S0[252*T])
    else:
        payoff_bs = final_price / np.mean(S0[252*T])

    print(f"Produit arrivé à maturité, payoff = {payoff_bs:.2f}")


# %%
import yfinance as yf
import numpy as np

stock = yf.download('MC.PA', start='2021-01-01', end='2025-04-25')

returns = np.log(stock['Close'] / stock['Close'].shift(1)).dropna()

volatility = returns.std() * np.sqrt(252)
volatility = float(volatility)

print(f"Volatilité historique annuelle entre 2021 et 2025 : {volatility:.2%}")
print(stock.iloc[0]["Close"])

# %%
