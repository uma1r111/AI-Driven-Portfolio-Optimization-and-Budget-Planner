import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

class BudgetCSP:
    def __init__(self, income, expenses, required_savings, risk_tolerance, curated_options=None, current_portfolio=None, step=50):
        self.income = income
        self.expenses = expenses
        self.required_savings = required_savings
        self.risk_tolerance = risk_tolerance.lower()
        self.current_portfolio = current_portfolio or {}
        self.step = step
        self.curated_options = curated_options or [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JNJ", "V", "PG",
            "MA", "HD", "DIS", "BAC", "XOM", "PFE", "KO", "WMT", "VZ", "CMCSA",
            "CRM", "INTC", "T", "CSCO", "CVX", "NKE", "ORCL", "ABT", "PEP", "MRK",
            "MCD", "ADBE", "ACN", "MDT", "TMO", "HON", "UNH", "BA", "ABBV", "AMAT",
            "COST", "DHR", "LIN", "QCOM", "IBM", "SBUX", "GE", "MMM", "UPS", "LMT",
            "CAT", "GS", "BLK", "AMGN", "SPGI", "TXN", "C", "ADP", "USB", "NOW",
            "MO", "CI", "TGT", "SCHW", "EQIX", "BKNG", "ISRG", "REGN", "BIIB", "FDX",
            "AXP", "DE", "DD", "EMR", "APD", "SHW", "FIS", "ADI", "STZ", "ZTS",
            "GILD", "ECL", "OXY", "SLB", "WFC", "PGR", "MET", "ICE", "ALL", "PNC",
            "RTX", "ANTM", "EW", "ITW", "EOG", "LOW", "CTAS", "LRCX", "SPG", "GM",
            "F", "NSC", "PLD", "AMT", "CCI", "DUK", "SO", "EXC", "AEP", "D", "SRE",
            "CMI", "HPQ", "BK", "APTV", "VLO", "FTNT", "CSX"
        ]
        self.investment_options = {t: {"max": 1000} for t in self.curated_options}
        current_investment = sum(self.current_portfolio.values())
        self._available_budget = income - expenses - required_savings - current_investment
        self.variables = {}
        self.constraints = []
        self.best_solution = None
        self.best_total = -1

    def risk_tolerance_percentage(self):
        return {"low": 0.25, "medium": 0.50, "high": 0.75}.get(self.risk_tolerance, 0.50)

    def update_investment_options_with_yfinance(self):
        """Update max allocations based on risk tolerance without price constraints"""
        risk_pct = self.risk_tolerance_percentage()
        print(f"\nUpdating {len(self.curated_options)} stocks (Risk: {self.risk_tolerance})")

        # Calculate max allocation based purely on risk tolerance
        base_max = int(self._available_budget * risk_pct)
        
        # Align to step size
        for stock in self.curated_options:
            step_max = (base_max // self.step) * self.step
            final_max = min(self.investment_options[stock]['max'], step_max)
            self.investment_options[stock]['max'] = final_max
            print(f"  {stock}: ${final_max}")

    def define_variables(self):
        """Create variables sorted by descending max allocation"""
        sorted_stocks = sorted(
            self.investment_options.items(),
            key=lambda x: x[1]['max'],
            reverse=True
        )
        
        for stock, info in sorted_stocks:
            max_invest = info['max']
            domain = list(range(0, max_invest + self.step, self.step))
            domain = [x for x in domain if x <= max_invest]
            self.variables[stock] = sorted(domain, reverse=True)

    def define_constraints(self):
        def total_constraint(assignment):
            return sum(assignment.values()) <= self._available_budget
        self.constraints.append(total_constraint)

    def is_consistent(self, assignment, var, value):
        temp = assignment.copy()
        temp[var] = value
        return all(c(temp) for c in self.constraints)

    def backtracking_search_optimal(self, assignment=None):
        assignment = assignment or {}
        
        if self.best_total == self._available_budget:
            return

        unassigned = [v for v in self.variables if v not in assignment]
        if not unassigned:
            current_total = sum(assignment.values())
            if current_total > self.best_total:
                self.best_total = current_total
                self.best_solution = assignment.copy()
                print(f"Found ${current_total}/{self._available_budget}")
            return

        next_var = unassigned[0]
        allocated = sum(assignment.values())
        
        for value in self.variables[next_var]:
            if (allocated + value) > self._available_budget:
                continue
                
            if self.is_consistent(assignment, next_var, value):
                assignment[next_var] = value
                self.backtracking_search_optimal(assignment)
                del assignment[next_var]
                
                if self.best_total == self._available_budget:
                    return

    def solve(self):
        if self._available_budget <= 0:
            return None
            
        self.update_investment_options_with_yfinance()
        self.define_variables()
        self.define_constraints()
        print(f"\nSearching {len(self.variables)} assets (Budget: ${self._available_budget})")
        self.backtracking_search_optimal()
        return self.best_solution

# Test Cases
def run_tests():
    print("\n====== Test 1: Full Allocation ======")
    csp1 = BudgetCSP(
        income=8000,
        expenses=5000,
        required_savings=1000,
        risk_tolerance="high",
        curated_options=["TSLA", "NVDA", "AMD"],
        current_portfolio={"AAPL": 500}
    )
    solution1 = csp1.solve()
    print("Optimal Allocation:", solution1)
    if solution1:
        total = sum(solution1.values())
        print(f"Budget Utilization: {total}/{csp1._available_budget} ({total/csp1._available_budget:.1%})")

    print("\n====== Test 2: Partial Allocation ======")
    csp2 = BudgetCSP(
        income=5000,
        expenses=4000,
        required_savings=800,
        risk_tolerance="medium",
        curated_options=["GOOGL", "AMZN"]
    )
    solution2 = csp2.solve()
    print("Best Solution:", solution2 or "No valid allocation")

def run_extended_tests():
    print("\n=== Test 4: Low Risk Allocation ===")
    csp4 = BudgetCSP(
        income=10000,
        expenses=6000,
        required_savings=2000,
        risk_tolerance="low",
        curated_options=["JNJ", "PG", "KO"],
        step=100
    )
    sol4 = csp4.solve()
    print("Conservative Allocation:", sol4)
    if sol4:
        total = sum(sol4.values())
        print(f"Used ${total} of ${csp4._available_budget}")

    print("\n=== Test 5: Small Step Size ===")
    csp5 = BudgetCSP(
        income=3000,
        expenses=2000,
        required_savings=500,
        risk_tolerance="medium",
        curated_options=["F", "GM"],
        step=10  # Testing granular increments
    )
    sol5 = csp5.solve()
    print("Precise Allocation:", sol5)
    
    print("\n=== Test 6: Single Stock Option ===")
    csp6 = BudgetCSP(
        income=7000,
        expenses=4000,
        required_savings=1000,
        risk_tolerance="high",
        curated_options=["TSLA"]
    )
    sol6 = csp6.solve()
    print("Single Stock Allocation:", sol6)

    print("\n=== Test 7: Mixed Price Stocks ===")
    csp7 = BudgetCSP(
        income=15000,
        expenses=8000,
        required_savings=3000,
        risk_tolerance="medium",
        curated_options=["BRK-A", "AMZN", "F"],  # Mix of high/low price stocks
        step=100
    )
    sol7 = csp7.solve()
    print("Mixed Price Allocation:", sol7)

    print("\n=== Test 8: Exact Budget Match ===")
    csp8 = BudgetCSP(
        income=5000,
        expenses=3200,
        required_savings=800,
        risk_tolerance="high",
        curated_options=["MSFT", "NVDA"],
        step=50
    )
    sol8 = csp8.solve()
    print("Exact Budget Solution:", sol8)

if __name__ == "__main__":
    run_tests()  # Original tests
    run_extended_tests()  # New tests
