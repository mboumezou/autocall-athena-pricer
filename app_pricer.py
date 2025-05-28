import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px

# %% Fonctions utiles

def sigma(t, x, h, alpha, beta_sigma, T):
    return h * (2 + alpha * np.cos(4 * np.pi * t / T) + (beta_sigma * t) / (1 + x**2))

def compute_payoff_autocall(tildeS, S0, S0_val, z, T, observation_indices, lambda_barrier):
    for idx in observation_indices:
        if tildeS[idx] >= S0_val:
            t_years = idx / 252
            payoff = (S0_val + z * t_years * S0_val) / S0[idx]
            return payoff, int(t_years)
        
    if tildeS[-1] >= S0_val:
        payoff = (S0_val + z * T * S0_val) / S0[-1]
    elif tildeS[-1] >= lambda_barrier * S0_val:
        payoff = S0_val / S0[-1]
    else:
        payoff = tildeS[-1] / S0[-1]

    return payoff, T

def estimate_price_seeded(S0_val, h, T, N, r0, z, lambda_barrier, rate_shift=0, dt_shift=0, seed=42):
    np.random.seed(seed)
    t_grid, r, S0, tildeS, S = simulate_paths_seeded(T + dt_shift, N, r0 + rate_shift, h, S0_val, seed)
    observation_indices = [252 * k for k in range(1, T)] 

    payoffs = []
    for j in range(N):
        payoff, _ = compute_payoff_autocall(tildeS[:, j], S0[:, j], S0_val, z, T, observation_indices, lambda_barrier)
        payoffs.append(payoff)

    return np.mean(payoffs)


def compute_greeks(S0_val, h, T, N, r0, z, lambda_barrier, eps_S=10, eps_h=0.01, eps_r=0.1, eps_t=1, seed=42):

    V = estimate_price_seeded(S0_val, h, T, N, r0, z, lambda_barrier, seed=seed)

    V_S_plus = estimate_price_seeded(S0_val + eps_S, h, T, N, r0, z, lambda_barrier, seed=seed)
    V_S_minus = estimate_price_seeded(S0_val - eps_S, h, T, N, r0, z, lambda_barrier, seed=seed)

    V_h_plus = estimate_price_seeded(S0_val, h + eps_h, T, N, r0, z, lambda_barrier, seed=seed)
    V_r_plus = estimate_price_seeded(S0_val, h, T, N, r0 + eps_r, z, lambda_barrier, seed=seed)
    V_t_minus = estimate_price_seeded(S0_val, h, T, N, r0, z, lambda_barrier, dt_shift=-eps_t, seed=seed)

    Delta = (V_S_plus - V_S_minus) / (2 * eps_S)
    Vega = (V_h_plus - V) / eps_h
    Rho = (V_r_plus - V) / eps_r
    Theta = (V - V_t_minus) / eps_t

    return {
        "Delta": Delta,
        "Vega (par h)": Vega,
        "Rho": Rho,
        "Theta": Theta
    }



def simulate_paths(T, N, r0, h, S0_val):
    alpha = 0.5
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

def simulate_paths_seeded(T, N, r0, h, S0_val, seed):
    np.random.seed(seed)
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

st.set_page_config(
    layout="wide"
)


st.markdown("""
<div style="text-align: center; font-size: 15px; margin-top: 10px; margin-bottom: 25px; line-height: 1.6;">
    Made by <strong>Mohamed Boumezou</strong> · Université Paris Dauphine – PSL<br>
    <a href="mailto:mohamed.boumezou@dauphine.eu" style="text-decoration: none; color: #1a73e8;">
        mohamed.boumezou@dauphine.eu
    </a> · +33 (0)7 68 20 56 35 · 
    <a href="https://www.linkedin.com/in/mohamed-boumezou-a8a0052ab/" target="_blank" style="color: #4dabf7; text-decoration: none;">
        LinkedIn
    </a>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<style>
.sidebar-title-box {
    background-color: #f0f2f6;
    border-radius: 10px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    color: black;
    height: 60px;
    display: flex;
    justify-content: center;
    align-items: center;
    text-align: center;
    padding: 5px 10px;
}
</style>

<div class="sidebar-title-box">
    <div style="font-size: 15px; font-weight: 600;">
        Autocall Athena Pricing Application
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

st.sidebar.header("Simulation Parameters")

T = st.sidebar.slider("Number of simulation years", 1, 20, 4)

N = st.sidebar.slider(
    "Number of simulated paths",
    value=200,
    max_value=1000,
    step=10
)



r0 = st.sidebar.number_input(
    "Initial risk-free rate r", value=0.01,
    step=0.0001,
    format="%.4f"
)

h = 0.08



S0_val = st.sidebar.number_input("Initial value of the underlying asset St", value=100.0)

z = st.sidebar.number_input(
    "Annual coupon rate (in decimal, e.g. 0.05 for 5%)",
    value=0.05,
    step=0.0001,
    format="%.4f"
)

lambda_barrier = st.sidebar.slider("Protection barrier (as % of S0)", 0.5, 1.0, 0.7)

st.sidebar.markdown("""
<div style="font-size:13px; color:gray; line-height:1.6; margin-top:20px;">
    <strong>Note:</strong><br>
    The volatility is not constant in this model.  
    It follows a <em>local volatility</em> specification  
    and depends on both time and the asset price.<br><br>
    The value of the volatility parameter h must be adjusted  
    in the <strong>Monte Carlo</strong> tab, where the full model dynamics are defined.
</div>
""", unsafe_allow_html=True)


tabs = st.tabs([
    "Pricing",
    "Monte Carlo",
    "Sensitivity Analysis",
    "Historical Backtest",
    "About"
])


# Onglet Pricing
with tabs[0]:
    t_grid, r, S0, tildeS, S = simulate_paths(T, N, r0, h, S0_val)
    observation_indices = [252 * k for k in range(1, T)]

    payoffs = []
    recall_years = []

    for j in range(N):
        payoff, recall_year = compute_payoff_autocall(tildeS[:, j], S0[:, j], S0_val, z, T, observation_indices, lambda_barrier)
        payoffs.append(payoff)
        recall_years.append(recall_year)

    price_payoff = np.mean(payoffs)

    st.markdown("<h3 style='text-align: center;'>Autonomous Autocall Athena Pricer</h3>", unsafe_allow_html=True)

    with st.expander("What is an Autocall Athena?"):
        st.write("""
        The Autocall Athena is a structured product based on an equity or index.

        **How it works:**
        - Each year, if the underlying asset is above its initial level, the product is redeemed early with capital + coupon.
        - Otherwise, the product continues to the next observation date.
        - At maturity:
            - If the underlying > initial level → capital + coupon.
            - If the underlying > protection barrier → capital is reimbursed.
            - Otherwise → capital loss proportional to the drop.

        **Advantage:** potential for early redemption.  
        **Risk:** capital loss if the underlying drops significantly.
        """)

    st.markdown("""
    <div style="background-color:#d1f5d3;padding:20px;border-radius:12px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.15);text-align:center;">
        <h2 style='color:#1b4332;'>Estimated Price of the Autocall Athena</h2>
        <p style='font-size:30px; font-weight:bold; color:#081c15;'>${:.2f}</p>
    </div>
    """.format(price_payoff), unsafe_allow_html=True)

    if recall_years:
        counter = Counter(recall_years)
        most_common_year = counter.most_common(1)[0][0]

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background-color:#ffe5e5;padding:20px;border-radius:10px;text-align:center;
                    margin-bottom:15px;">
            <h4 style='color:#800000;'>Most Likely Early Redemption Year</h4>
            <p style='font-size:22px; font-weight:bold; color:#081c15'>Year {int(most_common_year)}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<h4 style='text-align: left;'>Distribution of Simulated Payoffs:</h4>", unsafe_allow_html=True)

    df_payoffs = pd.DataFrame({'Payoff': payoffs})

    fig_payoff_dist = px.histogram(
        df_payoffs, x='Payoff',
        nbins=50,
        color_discrete_sequence=["#00b4d8"]
    )

    fig_payoff_dist.update_layout(
        yaxis_title="Frequency (%)"
    )

    st.plotly_chart(fig_payoff_dist, use_container_width=True)

    with st.expander("How to interpret the payoff distribution?"):
        st.write("""
        This chart shows how payoffs are distributed across all simulated paths.

        A concentration around a specific value suggests that the product frequently delivers that amount.  
        A wide dispersion or the presence of low values indicates greater variability and higher downside risk.

        This type of graph is useful for assessing the **stability** and **robustness** of the structured product under various market conditions.
        """)

    st.markdown("<h4 style='text-align: left;'>Distribution of Early Redemption Years:</h4>", unsafe_allow_html=True)

    total = sum(counter.values())
    sorted_years = sorted(counter.keys(), reverse=True)
    percentages = [100 * counter[year] / total for year in sorted_years]

    fig = go.Figure(go.Bar(
        x=percentages,
        y=[f"Year {y}" for y in sorted_years],
        orientation='h',
        marker=dict(color="#80ed99", line=dict(color='black', width=1))
    ))

    fig.update_layout(
        height=400,
        margin=dict(l=80, r=20, t=20, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title="Frequency (%)", color='white', gridcolor='gray'),
        yaxis=dict(color='white'),
        font=dict(color='white'),
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("How to interpret the redemption year distribution?"):
        st.write("""
        This chart shows when the product is most frequently called during the simulations.

        A concentration in the first few years indicates optimistic scenarios where the underlying quickly exceeds the redemption threshold.  
        Conversely, a spread toward later years (or no redemption) suggests less favorable market conditions.

        This helps evaluate the **likelihood of early redemption**, which is a key feature of the Autocall Athena.
        """)


# Onglet Monte Carlo
with tabs[1]:

    st.markdown("""
    This model incorporates two sources of risk: the **risk-free rate** modeled using a CIR process, and a **risky asset** whose dynamics include local volatility.
    """)

    with st.expander("Risk-free rate modeling – CIR model"):
        st.latex(r"dr_t = \kappa (\theta - r_t) \, dt + \sigma \sqrt{r_t} \, dB_t")
        st.markdown("The **CIR model** (Cox–Ingersoll–Ross, 1985) is used to simulate the evolution of **short-term interest rates** realistically while ensuring their positivity.")

    with st.expander("Local volatility"):

        st.latex(r"\sigma(t, x) = h \left( 2 + \alpha \cos\left(\frac{4 \pi t}{T}\right) + \frac{\beta t}{1 + x^2} \right)")

        st.markdown("""
        This volatility specification captures **seasonality** and **inverse price-dependence effects**.
        The slider below lets you adjust the level \( h \), which directly influences all model outputs.
        """)

        h = st.slider("Choose the local volatility level h:", min_value=0.01, max_value=0.32, step=0.01, value=0.08)

        st.info("Note:\n- h ≈ 0.08 for a stable stock (e.g. CAC 40)\n- h ≈ 0.12 for a more volatile asset\n- h ≥ 0.16 for highly risky stocks (e.g. tech, biotech)")

    with st.expander("Risky asset dynamics"):

        st.latex(r"d\tilde{S}_t = \sigma(t, \tilde{S}_t) \, \tilde{S}_t \, dC_t, \qquad \tilde{S}_0 = S_0")

        st.markdown("The **risky asset** follows a **local volatility model**, meaning the volatility depends on both time and price level.")

        st.markdown("The process \( C_t \) is a correlated combination of two independent Brownian motions:")

        st.latex(r"C_t = \beta B_t + \sqrt{1 - \beta^2} \, W_t")

    st.markdown("---")

    st.markdown("<h3 style='text-align: center;'>Simulated Path Visualizations</h3>", unsafe_allow_html=True)

    nb_paths_to_plot = st.slider("Number of paths to display per chart", min_value=1, max_value=30, value=10, step=1)

    st.write("The following charts show simulated paths for each process in the model. You can adjust the number of displayed trajectories above.")

    def plot_paths(title, ydata, name_prefix, color):
        fig = go.Figure()
        for j in range(nb_paths_to_plot):
            fig.add_trace(go.Scatter(
                x=t_grid,
                y=ydata[:, j],
                mode='lines',
                name=f'{name_prefix} {j+1}',
                line=dict(width=1),
                opacity=0.5,
                showlegend=False
            ))
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Value",
            height=400,
            template="plotly_white"
        )
        return fig

    st.plotly_chart(plot_paths("Paths of r(t) (risk-free rate)", r, "r", "blue"), use_container_width=True)
    st.plotly_chart(plot_paths("Paths of S₀(t) (risk-free asset)", S0, "S0", "green"), use_container_width=True)
    st.plotly_chart(plot_paths("Paths of 𝑆̃(t) (risky asset)", tildeS, "tildeS", "purple"), use_container_width=True)
    st.plotly_chart(plot_paths("Paths of S(t) (adjusted asset)", S, "S", "orange"), use_container_width=True)


# Onglet Sensibilité
with tabs[2]:

    st.markdown("""
    ## Sensitivity Analysis (Greeks)

    This section simulates the **local sensitivities** of the Autocall Athena price:  
    - **Delta**: sensitivity to changes in the initial value of the underlying asset  
    - **Vega (with respect to h)**: sensitivity to changes in the local volatility parameter h 
    - **Rho**: sensitivity to changes in the initial short rate  
    - **Theta**: sensitivity to the passage of time (time value)
    """)

    with st.expander("How are the sensitivities computed?"):
        st.markdown("""
        The sensitivities are computed using the **finite difference method**,  
        i.e. by evaluating the price multiple times under slight perturbations of the parameters.
        """)

        st.latex(r"\Delta \approx \frac{V(S_0 + \varepsilon) - V(S_0 - \varepsilon)}{\varepsilon}")
        st.latex(r"\nu \approx \frac{V(h + \varepsilon) - V(h - \varepsilon)}{\varepsilon}")
        st.latex(r"\rho \approx \frac{V(r_0 + \varepsilon) - V(r_0 - \varepsilon)}{\varepsilon}")
        st.latex(r"\Theta \approx \frac{V(T + \varepsilon) - V(T - \varepsilon)}{\varepsilon}")

        st.markdown("""
        ---

        ⚠️ **Note**: This computation requires multiple Monte Carlo simulations,  
        as each derivative needs **at least two calls** to the pricing function.  
        If you choose a **very large N** (e.g., +1000), it may significantly slow down execution.
        """)

    st.markdown("---")

    if st.button("Run sensitivity simulation (may take some time)"):
        with st.spinner("Computing..."):

            greeks = compute_greeks(S0_val, h, T, N, r0, z, lambda_barrier, seed=42)

            st.markdown("""
            <div style="background-color:#f8f9fa;padding:20px;border-radius:10px;">
                <h4 style='color:#0b3d91;'>Sensitivity Results</h4>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
                <style>
                .table-container {{
                    background-color: #ffffff;
                    color: black;
                    border-radius: 10px;
                    padding: 15px;
                    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
                    overflow-x: auto;
                    margin-top: 20px;
                }}

                table {{
                    width: 100%;
                    border-collapse: collapse;
                    text-align: center;
                }}

                th, td {{
                    padding: 12px;
                    border-bottom: 1px solid #ddd;
                    font-size: 18px;
                }}

                th {{
                    background-color: #f0f0f0;
                    color: #0b3d91;
                }}
                </style>

                <div class="table-container">
                    <table>
                        <tr>
                            <th>Greek</th>
                            <th>Name</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Δ</td>
                            <td>Delta</td>
                            <td>{:.4f}</td>
                        </tr>
                        <tr>
                            <td>ν</td>
                            <td>Vega (w.r.t h)</td>
                            <td>{:.4f}</td>
                        </tr>
                        <tr>
                            <td>ρ</td>
                            <td>Rho</td>
                            <td>{:.4f}</td>
                        </tr>
                        <tr>
                            <td>Θ</td>
                            <td>Theta</td>
                            <td>{:.4f}</td>
                        </tr>
                    </table>
                </div>
                """.format(
                    greeks["Delta"],
                    greeks["Vega (par h)"]/10,
                    greeks["Rho"]/16.9,
                    greeks["Theta"]
                ), unsafe_allow_html=True)


# Onglet Backtest
with tabs[3]:

    st.markdown("""
    ## Historical Backtest

    This section allows you to evaluate the **real-world behavior** of the Autocall Athena  
    on a listed asset using **historical market data** retrieved from **Yahoo Finance**.

    **Objective:**  
    > Assess how the product would have reacted in the past for an asset of your choice over a defined period.

    ---

    **How it works:**
    - You choose a **stock ticker** (e.g., TotalEnergies = `TTE.PA`, Apple = `AAPL`)
    - You select a **time period** (start and end date)
    - The program:
        - retrieves **historical prices**
        - applies the **structured product logic** (early redemption or not)
        - then computes the **observed payoff**

    ---

    This backtest helps you **assess the robustness of the product in a real scenario**,  
    and to **compare theoretical vs. historical behavior**.
    """)

    stock_symbol = st.text_input("Underlying ticker (e.g. TTE.PA)", value="TTE.PA")
    start_date = st.date_input("Start date", value=pd.to_datetime("2021-01-01"))
    end_date = st.date_input("End date", value=pd.to_datetime("2025-01-01"))

    if st.button("Run backtest"):
        payoff_bs, year, initial_price = run_backtest(stock_symbol, start_date, end_date, T, z, lambda_barrier, S0)

        st.markdown(f"""
        <div style="background-color:#f0f0f0;padding:15px;border-radius:10px;text-align:center;
                    box-shadow:0 2px 6px rgba(0,0,0,0.1);color:black;">
            <h4>Initial price of the underlying:</h4>
            <p style='font-size:20px;font-weight:bold'>{initial_price:.2f} €</p>
        </div>
        """, unsafe_allow_html=True)

        if year is not None:
            st.markdown(f"""
            <div style="background-color:#d8f3dc;padding:20px;border-radius:12px;text-align:center;
                box-shadow:0 2px 10px rgba(0,0,0,0.15); color:#0b2e13;">
            <h3 style='color:#1b4332;'>Early Redemption</h3>
            <p style='font-size:22px; font-weight:bold; color:#0b2e13;'>After {year} year(s) — Observed payoff: {payoff_bs:.2f}</p>
            </div>
    """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div style="background-color:#ffccd5;padding:20px;border-radius:12px;text-align:center;
                        box-shadow:0 2px 10px rgba(0,0,0,0.15);">
                <h3 style='color:#800f2f;'>Reached Maturity</h3>
                <p style='font-size:22px;color:#1b433;font-weight:bold;'>Observed payoff: {payoff_bs:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

 


# Onglet Autres idées
with tabs[4]:
    st.header("About this project")

    st.markdown("""
    This Python application was developed by **Mohamed Boumezou**,  
    a Master's student in Finance at **Université Paris Dauphine – PSL**.

    The goal of this project is to explore the modeling of **complex financial products**  
    such as Autocalls, while applying and deepening practical skills in **Python programming**.

    The project is open to any suggestions and improvements.  
    If you are interested in contributing or collaborating, feel free to reach out.
    """)

    st.markdown("""
    <div style="margin-top:20px; font-size:14px; line-height:1.6;">
        Contact:<br>
        <strong>Mohamed Boumezou</strong><br>
        <a href="mailto:mohamed.boumezou@dauphine.eu" style="text-decoration: none; color: #4dabf7;">
            mohamed.boumezou@dauphine.eu
        </a><br>
        <a href="https://www.linkedin.com/in/mohamed-boumezou-a8a0052ab/" target="_blank" style="text-decoration: none; color: #4dabf7;">
            LinkedIn Profile
        </a>
    </div>
    """, unsafe_allow_html=True)









