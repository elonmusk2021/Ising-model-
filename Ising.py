import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. 數據收集與預處理
def generate_market_data(n_assets=10, n_days=1000):
    np.random.seed(42)
    returns = np.random.randn(n_days, n_assets) * 0.02  # 模擬每日收益率
    prices = 100 + np.cumsum(returns, axis=0)  # 生成價格數據
    return pd.DataFrame(prices, columns=[f"Asset_{i}" for i in range(n_assets)])

data = generate_market_data()
returns = data.pct_change().dropna()

# 加權自旋：根據收益率絕對值計算影響力
weights = np.abs(returns).mean().values  # 每個資產的平均絕對收益率作為加權，形狀 (10,)

# 2. 計算交互矩陣 J_ij
corr_matrix = np.corrcoef(returns.T)  # 用原始收益率計算相關性
J = -corr_matrix  # 取負號，高相關性意味著強相互作用

# 3. Ising 模型 Metropolis-Hastings 演算法（考慮加權自旋）
def metropolis_hastings(J, spins, weights, beta=1.0, steps=1000):
    n = len(spins)
    weighted_spins = spins * weights  # 初始加權自旋，形狀 (10,)
    for _ in range(steps):
        i = np.random.randint(0, n)
        # 能量變化考慮加權自旋
        delta_E = 2 * spins[i] * weights[i] * np.sum(J[i] * weighted_spins)
        if delta_E < 0 or np.random.rand() < np.exp(-beta * delta_E):
            spins[i] *= -1  # 翻轉自旋
            weighted_spins[i] = spins[i] * weights[i]  # 更新加權自旋
    return spins, weighted_spins

# 初始化 Ising 模型
n_assets = returns.shape[1]
spins = np.random.choice([-1, 1], size=n_assets)  # 初始自旋方向，形狀 (10,)
spins, weighted_spins = metropolis_hastings(J, spins, weights)

# 4. 交易訊號生成（基於加權自旋）
def generate_signals(weighted_spins, threshold=0.7):
    mean_weighted_spin = np.mean(weighted_spins) / np.mean(weights)  # 正規化平均加權自旋
    if mean_weighted_spin > threshold:
        return "BUY"
    elif mean_weighted_spin < -threshold:
        return "SELL"
    else:
        return "HOLD"

signal = generate_signals(weighted_spins)
print(f"trade signal: {signal}")

# 5. 簡單回測
def backtest(data, signal, initial_cash=10000):
    cash = initial_cash
    position = 0
    for i in range(len(data) - 1):
        if signal == "BUY" and cash > 0:
            position = cash / data.iloc[i]
            cash = 0
        elif signal == "SELL" and position > 0:
            cash = position * data.iloc[i]
            position = 0
    final_value = cash + (position * data.iloc[-1]) if position > 0 else cash
    return final_value

final_portfolio_value = backtest(data["Asset_0"], signal)
print(f"final value: {final_portfolio_value:.2f}")

# 6. 視覺化市場動態與加權自旋結果
plt.figure(figsize=(12, 5))
plt.plot(data.index, data["Asset_0"], label="Asset Price")
plt.title("Market Simulation with Weighted Ising Model Signals")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()

# 額外輸出：檢查加權自旋
print("weight spin:", weighted_spins)
print("avg weight spin:", np.mean(weighted_spins))