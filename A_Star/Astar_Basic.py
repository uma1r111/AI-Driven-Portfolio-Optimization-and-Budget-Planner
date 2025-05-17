import heapq
import itertools
import numpy as np
from csp_v3 import BudgetCSP  # Import from separate file

class AStarPortfolioOptimizer:
    def __init__(self, initial_state, budget_csp, stock_predictions, step=50):
        self.initial_state = initial_state
        self.budget_csp = budget_csp
        self.predictions = stock_predictions
        self.step = step

    def g(self, state):
        """Actual cost: negative expected returns"""
        return -sum(self.predictions.get(stock, 0) * amount 
                   for stock, amount in state.items())

    def h(self, state):
        """Heuristic: potential from remaining budget"""
        invested = sum(state.values())
        remaining = self.budget_csp._available_budget - invested
        max_rate = max(self.predictions.values()) if self.predictions else 0
        return - (remaining * max_rate)

    def f(self, state):
        return self.g(state) + self.h(state)

    def is_valid(self, state):
        return all(constraint(state) for constraint in self.budget_csp.constraints)

    def neighbors(self, state):
        neighbors = []
        stocks = list(state.keys())
        
        # Single-stock moves
        for stock in stocks:
            for delta in (-self.step, self.step):
                new_state = state.copy()
                new_state[stock] = max(0, new_state[stock] + delta)
                if self.budget_csp.is_consistent(new_state, stock, new_state[stock]):
                    neighbors.append(new_state)
        
        # Pairwise transfers
        for i, j in itertools.combinations(stocks, 2):
            for delta in (self.step, -self.step):
                new_state = state.copy()
                new_state[i] = max(0, new_state[i] + delta)
                new_state[j] = max(0, new_state[j] - delta)
                if (self.budget_csp.is_consistent(new_state, i, new_state[i]) and 
                    self.budget_csp.is_consistent(new_state, j, new_state[j])):
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

if __name__ == "__main__":
    from csp_v3 import BudgetCSP

    csp = BudgetCSP(
        income=10000,
        expenses=4000,
        required_savings=1000,
        risk_tolerance="medium",
        curated_options=["AAPL", "MSFT", "GOOGL"],
        step=50,
        min_stocks=2
    )

    #Call solve() to set constraints and budget
    csp.solve()

    # Hardcoded predictions (OK for now)
    stock_predictions = {
        "AAPL": 0.07,
        "MSFT": 0.06,
        "GOOGL": 0.05
    }

    optimizer = AStarPortfolioOptimizer(
        initial_state={t: 0 for t in stock_predictions},
        budget_csp=csp,
        stock_predictions=stock_predictions,
        step=50
    )

    result = optimizer.search()
    print("Optimal Portfolio:", result)
    if result:
      total = sum(result.values())
      expected = sum(stock_predictions[t] * result[t] for t in result)
      print(f"\nTotal Investment: ${total}")
      print(f"Expected Return: ${expected:.2f}")

