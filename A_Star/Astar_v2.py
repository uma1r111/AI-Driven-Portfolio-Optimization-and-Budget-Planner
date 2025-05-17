import pandas as pd
import heapq
import itertools
from csp_v4 import BudgetCSP

# -------------------------------
# Step 1: Load Heuristics Only
# -------------------------------
def load_heuristics(top_stocks):
    try:
        df = pd.read_csv("Stocks - Heuristics.csv")
        stock_heuristics = {}

        for _, row in df.iterrows():
            stock = row['Stock']
            if stock in top_stocks:
                stock_heuristics[stock] = row['Best Heuristic']

        return stock_heuristics
    except Exception as e:
        print(f"Error reading Stocks - Heuristics.csv: {e}")
        return {}

# -------------------------------
# Step 2: A* Portfolio Optimizer
# -------------------------------
class AStarPortfolioOptimizer:
    def __init__(self, initial_state, budget_csp, heuristics, step=50):
        self.initial_state = initial_state
        self.budget_csp = budget_csp
        self.heuristics = heuristics
        self.step = step
        # Track the number of iterations
        self.iterations = 0
        self.max_iterations = 10000  # Prevent infinite loops
        
        # Create a more diversified initial state to help with min_stocks constraint
        # This helps the search start from a more promising position
        self.diversified_initial_state = self.create_diversified_initial_state()

    def create_diversified_initial_state(self):
        """Create an initial state that already satisfies the min_stocks constraint"""
        state = {t: 0 for t in self.initial_state}
        
        # Sort stocks by their heuristic value (descending)
        sorted_stocks = sorted(self.heuristics.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate equally distributed budget
        stocks_to_include = min(len(sorted_stocks), self.budget_csp.min_stocks)
        budget_per_stock = self.budget_csp._available_budget / stocks_to_include
        budget_per_stock = (budget_per_stock // self.step) * self.step  # Round to nearest step
        
        # Distribute budget across top stocks
        remaining_budget = self.budget_csp._available_budget
        for i in range(stocks_to_include):
            if remaining_budget <= 0:
                break
                
            stock = sorted_stocks[i][0]
            max_alloc = min(self.budget_csp.investment_options[stock]['max'], budget_per_stock)
            max_alloc = (max_alloc // self.step) * self.step  # Round to nearest step
            
            if max_alloc > 0:
                state[stock] = max_alloc
                remaining_budget -= max_alloc
        
        # Verify if it meets constraints
        if self.is_valid(state):
            print(f"Created valid diversified initial state: {state}")
            print(f"Stocks with allocation: {self.count_stocks_with_allocation(state)}")
            print(f"Total invested: ${sum(state.values())}")
            return state
            
        # If not valid, return original empty state
        return self.initial_state

    def construct_greedy_solution(self):
        """Construct a greedy solution that satisfies the min_stocks constraint"""
        state = {t: 0 for t in self.initial_state}
        
        # Sort stocks by their heuristic value (descending)
        sorted_stocks = sorted(self.heuristics.items(), key=lambda x: x[1], reverse=True)
        
        # Ensure we have min_stocks with allocation
        min_required = self.budget_csp.min_stocks
        stocks_to_try = sorted_stocks[:min_required]
        
        # Start with min allocation for each required stock (step size)
        remaining_budget = self.budget_csp._available_budget
        for stock, _ in stocks_to_try:
            state[stock] = self.step
            remaining_budget -= self.step
            
        if remaining_budget < 0:
            return None  # Not enough budget for minimum allocation
            
        # Now distribute remaining budget according to heuristics
        for idx, (stock, heuristic) in enumerate(sorted_stocks):
            # Only allow additional allocation if budget remains
            if remaining_budget <= 0:
                break
                
            # Calculate maximum additional allocation
            max_additional = min(
                remaining_budget,
                self.budget_csp.investment_options[stock]['max'] - state.get(stock, 0)
            )
            max_additional = (max_additional // self.step) * self.step
            
            if max_additional > 0:
                state[stock] = state.get(stock, 0) + max_additional
                remaining_budget -= max_additional
                
        # Check if this satisfies all constraints
        if self.is_valid(state):
            return state
        
        # If not valid, try a more conservative approach
        state = {t: 0 for t in self.initial_state}
        for i in range(min_required):
            if i < len(sorted_stocks):
                stock = sorted_stocks[i][0]
                state[stock] = self.step
        
        return state if self.is_valid(state) else None

    def g(self, state):
        invested = sum(state.values())
        unused_penalty = 0.0001 * (self.budget_csp._available_budget - invested)
        return -sum(self.heuristics.get(stock, 0) * amount for stock, amount in state.items()) + unused_penalty

    def h(self, state):
        invested = sum(state.values())
        remaining_budget = self.budget_csp._available_budget - invested
        if remaining_budget <= 0:
            return 0

        estimated_gain = 0
        for stock in self.heuristics:
            if stock in state:
                allocated = state[stock]
                max_alloc = self.budget_csp.investment_options[stock]['max']
                if allocated < max_alloc:
                    estimated_gain += self.heuristics[stock]

        return -estimated_gain

    def f(self, state):
        return self.g(state) + self.h(state)

    def is_valid(self, state):
        return all(constraint(state) for constraint in self.budget_csp.constraints)

    def count_stocks_with_allocation(self, state):
        """Count how many stocks have a positive allocation"""
        return sum(1 for amount in state.values() if amount > 0)

    def neighbors(self, state):
        neighbors = []
        stocks = list(state.keys())
        
        # First, ensure we have enough stocks with allocation
        current_stock_count = self.count_stocks_with_allocation(state)
        
        # IMPROVED: Special handling for diversification constraint
        if current_stock_count < self.budget_csp.min_stocks:
            # First, try to allocate to stocks that currently have zero allocation
            zero_stocks = [stock for stock in stocks if state[stock] == 0]
            
            # Start with minimum allocation to satisfy diversification
            for stock in zero_stocks:
                new_state = state.copy()
                new_state[stock] = self.step  # Minimum allocation
                
                # Check total budget after this allocation
                new_total = sum(new_state.values())
                
                if new_total <= self.budget_csp._available_budget and self.budget_csp.is_consistent(new_state, stock, new_state[stock]):
                    neighbors.append(new_state)
                    
            # If we're still struggling to meet the min_stocks requirement,
            # try reallocating from stocks with higher allocations
            if len(neighbors) == 0 and current_stock_count > 0:
                # Find stocks with allocations
                allocated_stocks = [(s, amt) for s, amt in state.items() if amt > 0]
                allocated_stocks.sort(key=lambda x: x[1], reverse=True)  # Highest allocation first
                
                # Try to take from highest allocated stock to give to zero stocks
                if allocated_stocks:
                    donor_stock, donor_amount = allocated_stocks[0]
                    
                    for zero_stock in zero_stocks:
                        new_state = state.copy()
                        
                        # Take step size from donor, give to zero stock
                        if donor_amount >= self.step * 2:
                            new_state[donor_stock] = donor_amount - self.step
                            new_state[zero_stock] = self.step
                            
                            if self.budget_csp.is_consistent(new_state, donor_stock, new_state[donor_stock]) and \
                               self.budget_csp.is_consistent(new_state, zero_stock, new_state[zero_stock]):
                                neighbors.append(new_state)
        
        # Standard neighbors generation for all stocks
        for stock in stocks:
            max_alloc = self.budget_csp.investment_options[stock]['max']
            
            # Generate more potential states with finer granularity
            if state[stock] == 0:
                # Add more options for increasing from zero, especially important for min_stocks
                for alloc in [self.step, self.step*2, self.step*4]:
                    if alloc <= max_alloc:
                        new_state = state.copy()
                        new_state[stock] = alloc
                        new_total = sum(new_state.values())
                        
                        if new_total <= self.budget_csp._available_budget and self.budget_csp.is_consistent(new_state, stock, new_state[stock]):
                            neighbors.append(new_state)
            else:
                # For stocks that already have allocation
                for delta in [-self.step*2, -self.step, self.step, self.step*2]:
                    new_state = state.copy()
                    new_amount = max(0, min(new_state[stock] + delta, max_alloc))
                    
                    # Skip if no change
                    if new_amount == new_state[stock]:
                        continue
                        
                    new_state[stock] = new_amount
                    new_total = sum(new_state.values())
                    
                    if new_total <= self.budget_csp._available_budget and self.budget_csp.is_consistent(new_state, stock, new_state[stock]):
                        neighbors.append(new_state)
        
        # IMPROVED: Pairwise rebalancing for better exploration
        # This is critical for finding diverse solutions
        if len(stocks) >= 2:
            for i, stock1 in enumerate(stocks):
                for stock2 in stocks[i+1:]:
                    # Transfer from stock1 to stock2
                    if state[stock1] >= self.step:
                        new_state = state.copy()
                        new_state[stock1] -= self.step
                        
                        # Avoid exceeding max allocation
                        if state[stock2] + self.step <= self.budget_csp.investment_options[stock2]['max']:
                            new_state[stock2] += self.step
                            
                            if self.budget_csp.is_consistent(new_state, stock1, new_state[stock1]) and \
                               self.budget_csp.is_consistent(new_state, stock2, new_state[stock2]):
                                neighbors.append(new_state)
                    
                    # Transfer from stock2 to stock1
                    if state[stock2] >= self.step:
                        new_state = state.copy()
                        new_state[stock2] -= self.step
                        
                        # Avoid exceeding max allocation
                        if state[stock1] + self.step <= self.budget_csp.investment_options[stock1]['max']:
                            new_state[stock1] += self.step
                            
                            if self.budget_csp.is_consistent(new_state, stock1, new_state[stock1]) and \
                               self.budget_csp.is_consistent(new_state, stock2, new_state[stock2]):
                                neighbors.append(new_state)
                                
        return neighbors

    def search(self):
        open_set = []
        counter = itertools.count()
        
        # Try to start with a diversified initial state
        start_state = self.diversified_initial_state
        
        heapq.heappush(open_set, (self.f(start_state), next(counter), start_state))
        visited = set()
        best_valid_state = None
        best_valid_score = float('inf')

        while open_set and self.iterations < self.max_iterations:
            self.iterations += 1
            current_f, _, current = heapq.heappop(open_set)
            current_key = frozenset(current.items())

            if current_key in visited:
                continue
                
            visited.add(current_key)

            # Check if this state satisfies all constraints
            if self.is_valid(current):
                current_score = self.f(current)
                if current_score < best_valid_score:
                    best_valid_score = current_score
                    best_valid_state = current
                    print(f"Found valid portfolio (iteration {self.iterations}) with score {best_valid_score}")
                    print(f"  Allocations: {current}")
                    print(f"  Stocks with allocation: {self.count_stocks_with_allocation(current)}")
                    print(f"  Total invested: ${sum(current.values())}")

            neighbors = self.neighbors(current)
            for neighbor in neighbors:
                neighbor_key = frozenset(neighbor.items())
                if neighbor_key not in visited:
                    heapq.heappush(open_set, (self.f(neighbor), next(counter), neighbor))

            # Print progress updates
            if self.iterations % 1000 == 0:
                print(f"Iteration {self.iterations}, explored {len(visited)} states, queue size: {len(open_set)}")

        if self.iterations >= self.max_iterations:
            print(f"Warning: Reached maximum iterations ({self.max_iterations})")
            
        # If no optimal solution found, try to construct a greedy solution
        if best_valid_state is None:
            print("No optimal solution found. Attempting to construct a greedy diversified solution...")
            greedy_solution = self.construct_greedy_solution()
            if greedy_solution and self.is_valid(greedy_solution):
                best_valid_state = greedy_solution
                print(f"Created greedy solution: {greedy_solution}")
                print(f"  Stocks with allocation: {self.count_stocks_with_allocation(greedy_solution)}")
                print(f"  Total invested: ${sum(greedy_solution.values())}")
            
        return best_valid_state

# -------------------------------
# Step 3: Main Program
# -------------------------------
if __name__ == "__main__":
    # Top stocks we want to optimize
    top_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

    # Load only heuristic values
    heuristics = load_heuristics(top_stocks)
    print("\nHeuristic values (Best Heuristic):")
    print(heuristics)

    if not heuristics:
        print("\nNo matching stocks found! Check your CSV file or stock names.")
        exit()

    # Step 2: Setup CSP
    csp = BudgetCSP(
        income=10000,
        expenses=4000,
        required_savings=1000,
        risk_tolerance="high",
        curated_options=list(heuristics.keys()),
        step=50,
        min_stocks=3  # Now can handle more than 1 stock minimum
    )
    csp.update_investment_options_with_yfinance()
    csp.define_variables()
    csp.define_constraints()
    
    # Create a manually constructed valid portfolio for comparison
    # This ensures we have at least one valid solution to compare against
    manual_portfolio = {}
    sorted_stocks = sorted(heuristics.items(), key=lambda x: x[1], reverse=True)
    budget_per_stock = csp._available_budget / csp.min_stocks
    budget_per_stock = (budget_per_stock // csp.step) * csp.step
    
    remaining_budget = csp._available_budget
    for i in range(min(len(sorted_stocks), csp.min_stocks)):
        stock = sorted_stocks[i][0]
        max_alloc = min(csp.investment_options[stock]['max'], budget_per_stock)
        max_alloc = (max_alloc // csp.step) * csp.step
        manual_portfolio[stock] = max_alloc
        remaining_budget -= max_alloc
        
    print("\nManual reference portfolio:")
    print(manual_portfolio)
    print(f"Total: ${sum(manual_portfolio.values())}")
    print(f"Stocks with allocation: {sum(1 for v in manual_portfolio.values() if v > 0)}")
    
    # Check if manual portfolio is valid
    valid = all(constraint(manual_portfolio) for constraint in csp.constraints)
    print(f"Is valid: {valid}")

    # Step 3: A* Optimization
    optimizer = AStarPortfolioOptimizer(
        initial_state={t: 0 for t in heuristics},
        budget_csp=csp,
        heuristics=heuristics,
        step=50
    )

    result = optimizer.search()

    print("\nOptimal Portfolio:", result)
    if result:
        total = sum(result.values())
        expected_score = sum(heuristics[t] * result[t] for t in result)
        print(f"\nTotal Investment: ${total}")
        print(f"Expected Heuristic Score: {expected_score:.2f}")
        print(f"Number of stocks in portfolio: {optimizer.count_stocks_with_allocation(result)}")
    else:
        print("\nNo valid portfolio found. Returning manual reference portfolio.")
        result = manual_portfolio
        total = sum(result.values())
        expected_score = sum(heuristics.get(t, 0) * result[t] for t in result)
        print(f"\nTotal Investment: ${total}")
        print(f"Expected Heuristic Score: {expected_score:.2f}")
        print(f"Number of stocks in portfolio: {sum(1 for v in result.values() if v > 0)}")