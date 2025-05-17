# run_optimizer_main.py
import pandas as pd
import numpy as np
from csp_v3 import BudgetCSP, TICKER_TO_NAME, fetch_data_cached
from Astar_v3 import AStarPortfolioOptimizer, load_heuristics
import random

# -------------------------------
# User Selected Stocks (Frontend Selection)
# -------------------------------
user_selected_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]

# Generate 3 random diverse combinations from user-selected stocks
random.shuffle(user_selected_stocks)
stock_combinations = [
    user_selected_stocks[:3],
    user_selected_stocks[1:4],
    user_selected_stocks[2:5]
]

user_income = 12000
user_expenses = 5000
user_savings = 2000
user_risk = "medium"
step_size = 50
min_stock_count = 3

summary_data = []
portfolio_advice = []

for i, top_stocks in enumerate(stock_combinations, 1):
    print(f"\n===================== Portfolio {i} =====================")
    print(f"Stocks Considered: {top_stocks}")

    heuristics = load_heuristics(top_stocks)
    if not heuristics:
        print("No valid heuristics found. Skipping.")
        continue

    csp = BudgetCSP(
        income=user_income,
        expenses=user_expenses,
        required_savings=user_savings,
        risk_tolerance=user_risk,
        curated_options=list(heuristics.keys()),
        step=step_size,
        min_stocks=min_stock_count
    )
    csp.update_investment_options_with_yfinance()
    csp.define_variables()
    csp.define_constraints()

    optimizer = AStarPortfolioOptimizer(
        initial_state={t: 0 for t in heuristics},
        budget_csp=csp,
        heuristics=heuristics,
        step=step_size
    )

    final_portfolio = optimizer.search()

    if not final_portfolio:
        print("⚠️ No valid portfolio found!")
        continue

    total_investment = sum(final_portfolio.values())
    total_score = sum(heuristics.get(stock, 0) * amount for stock, amount in final_portfolio.items())
    average_score_per_dollar = total_score / total_investment if total_investment > 0 else 0
    num_stocks_used = sum(1 for v in final_portfolio.values() if v > 0)

    # Financial metrics
    try:
        data = fetch_data_cached(tuple(top_stocks), period="60d", auto_adjust=True, column='Close')
        returns = data.pct_change().dropna()
        cov_matrix = returns.cov() * 252
        weights = np.array([final_portfolio.get(stock, 0) / total_investment for stock in top_stocks])
        port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        port_std_dev = np.sqrt(port_variance)
        port_return = (returns.mean() * 252).dot(weights)
        risk_free_rate = 0.01
        sharpe_ratio = (port_return - risk_free_rate) / port_std_dev if port_std_dev != 0 else 0
    except Exception as e:
        port_variance = 0
        port_return = 0
        port_std_dev = 0
        sharpe_ratio = 0
        print(f"⚠️ Financial metric calculation failed due to: {e}")

    portfolio_df = pd.DataFrame([
        {
            "Stock": stock,
            "Company Name": TICKER_TO_NAME.get(stock, "Unknown"),
            "Allocation ($)": amount,
            "Heuristic Score": round(heuristics.get(stock, 0), 4)
        }
        for stock, amount in final_portfolio.items() if amount > 0
    ])
    portfolio_df["% of Total"] = round(100 * portfolio_df["Allocation ($)"] / total_investment, 2)
    portfolio_df = portfolio_df.sort_values(by="Allocation ($)", ascending=False)

    print("\n✅ Portfolio Optimization Results:")
    print(portfolio_df.to_string(index=False))
    print(f"\nExpected Heuristic Score (weighted): {total_score:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.3f} | Expected Return: {port_return:.3%} | Volatility: {port_std_dev:.3%}")
    print("========================================================\n")

    summary_data.append({
        "Portfolio": i,
        "Stocks Used": num_stocks_used,
        "Total Investment": total_investment,
        "Total Score": round(total_score, 2),
        "Avg Score/$": round(average_score_per_dollar, 4),
        "Sharpe Ratio": round(sharpe_ratio, 3),
        "Return": round(port_return * 100, 2),
        "Volatility": round(port_std_dev * 100, 2),
        "Included Stocks": ", ".join([s for s, v in final_portfolio.items() if v > 0])
    })

    portfolio_advice.append((i, total_score, average_score_per_dollar, sharpe_ratio))

# -------------------------------
# Final Summary Table and Advice
# -------------------------------
if summary_data:
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values(by="Total Score", ascending=False)

    print("\n===================== Summary of All Portfolios =====================")
    print(summary_df.to_string(index=False))

    best = max(portfolio_advice, key=lambda x: (x[1], x[2], x[3]))
    print(f"\n💡 Recommended Portfolio: Portfolio {best[0]}")
    print("Reason: Highest heuristic return, efficiency, and financial quality (Sharpe ratio).")
    print("\n📌 Guidance:")
    print("- If you're a conservative investor, look at Sharpe Ratio and Volatility.")
    print("- If you want growth, maximize Heuristic Score.")
    print("- If you're uncertain, choose the highest Avg Score/$.\n")
else:
    print("\n❌ No valid portfolios were generated. Try selecting more stocks or relaxing constraints like 'min_stock_count'.")
