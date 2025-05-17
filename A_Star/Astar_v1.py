import pandas as pd
import heapq
import itertools
from csp_v3 import BudgetCSP

# -------------------------------
# Step 1: Load Sector CSVs
# -------------------------------
def load_predictions(top_stocks):
    # List of your sector CSV filenames
    sector_files = [
        "technology_sector_predictions.csv",
        "PhysicalAssets_sector_predictions.csv",
        "Industrial_sector_predictions.csv",
        "healthcare_sector_predictions.csv",
        "FinancialService_sector_predictions.csv",
        "ConsumerDefence_sector_predictions.csv",
        "ConsumerCyclic_sector_predictions.csv",
        "CommunicationServices_sector_predictions.csv"
    ]

    stock_predictions = {}

    for file in sector_files:
        try:
            df = pd.read_csv(file)
            for _, row in df.iterrows():
                stock = row['Stock']
                if stock in top_stocks:
                    stock_predictions[stock] = row['Log_Returns']
        except Exception as e:
            print(f"Error reading {file}: {e}")

    return stock_predictions

# -------------------------------
# Step 2: A* Portfolio Optimizer
# -------------------------------
class AStarPortfolioOptimizer:
    def __init__(self, initial_state, budget_csp, stock_predictions, step=50):
        self.initial_state = initial_state
        self.budget_csp = budget_csp
        self.predictions = stock_predictions
        self.step = step

    def g(self, state):
        """Actual cost: negative expected returns + unused budget penalty"""
        invested = sum(state.values())
        unused_penalty = 0.0001 * (self.budget_csp._available_budget - invested)
        return -sum(self.predictions.get(stock, 0) * amount for stock, amount in state.items()) + unused_penalty

    def h(self, state):
       invested = sum(state.values())
       remaining_budget = self.budget_csp._available_budget - invested
       if remaining_budget <= 0:
         return 0

    # Only consider stocks where some budget can still go
       candidates = []
       for stock, prediction in self.predictions.items():
           allocated = state.get(stock, 0)
           max_alloc = self.budget_csp.investment_options[stock]['max']
           if allocated < max_alloc:
             candidates.append((stock, prediction))

    # Sort remaining candidates
       sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

       if not sorted_candidates:
        return 0

       top_k = min(4, len(sorted_candidates))  # Top 2 or less
       top_returns = sum(rate for stock, rate in sorted_candidates[:top_k])

       estimated_gain = remaining_budget * (top_returns / top_k)
       return -estimated_gain



    def f(self, state):
        return self.g(state) + self.h(state)

    def is_valid(self, state):
        return all(constraint(state) for constraint in self.budget_csp.constraints)

    def neighbors(self, state):
        neighbors = []
        stocks = list(state.keys())

        for stock in stocks:
            max_alloc = self.budget_csp.investment_options[stock]['max']
            for delta in range(-self.step * 20, self.step * 21, self.step):
                if delta == 0:
                   continue
                new_state = state.copy()
                new_amount = max(0, new_state[stock] + delta)
                new_amount = min(new_amount, max_alloc)
                new_state[stock] = new_amount
                if self.budget_csp.is_consistent(new_state, stock, new_state[stock]):
                   neighbors.append(new_state)

        return neighbors


    def search(self):
        open_set = []
        counter = itertools.count()
        heapq.heappush(open_set, (self.f(self.initial_state), next(counter), self.initial_state))
        visited = set()

        while open_set:
            current_f, _, current = heapq.heappop(open_set)
            current_key = frozenset(current.items())

            if current_key in visited:
                continue
            visited.add(current_key)

            if self.is_valid(current):
                return current

            for neighbor in self.neighbors(current):
                neighbor_key = frozenset(neighbor.items())
                if neighbor_key not in visited:
                    heapq.heappush(open_set, (self.f(neighbor), next(counter), neighbor))

        return None

# -------------------------------
# Step 3: Main Program
# -------------------------------
if __name__ == "__main__":
    # Top stocks we want to optimize
    top_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

    # Step 1: Load LSTM predictions
    stock_predictions = load_predictions(top_stocks)
    print("\nPredicted returns (Log_Returns):")
    print(stock_predictions)
        # TEMP FIX: Scale predictions for small returns
    for stock in stock_predictions:
        stock_predictions[stock] *= 10


    if not stock_predictions:
        print("\nNo matching stocks found! Check your CSVs or stock names.")
        exit()

    # Step 2: Setup CSP
    csp = BudgetCSP(
        income=10000,
        expenses=4000,
        required_savings=1000,
        risk_tolerance="medium",
        curated_options=list(stock_predictions.keys()),
        step=50,
        min_stocks=1
    )
    csp.update_investment_options_with_yfinance()
    csp.define_variables()
    csp.define_constraints()


    # Step 3: A* Optimization
    optimizer = AStarPortfolioOptimizer(
        initial_state={t: 0 for t in stock_predictions},
        budget_csp=csp,
        stock_predictions=stock_predictions,
        step=50
    )

    result = optimizer.search()

    print("\nOptimal Portfolio:", result)
    if result:
        total = sum(result.values())
        expected_return = sum(stock_predictions[t] * result[t] for t in result)
        print(f"\nTotal Investment: ${total}")
        print(f"Expected Return: ${expected_return:.2f}")
