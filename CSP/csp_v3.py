import yfinance as yf
import numpy as np
import pandas as pd
import time
from scipy.optimize import minimize
from functools import lru_cache

#############################################
# Global Mapping: Ticker to Company Names
#############################################
TICKER_TO_NAME = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp.",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "TSLA": "Tesla Inc.",
    "META": "Meta Platforms Inc.",
    "NVDA": "NVIDIA Corp.",
    "JNJ": "Johnson & Johnson",
    "V": "Visa Inc.",
    "PG": "Procter & Gamble",
    "MA": "Mastercard Inc.",
    "HD": "Home Depot Inc.",
    "DIS": "Walt Disney Co.",
    "BAC": "Bank of America Corp.",
    "XOM": "Exxon Mobil Corp.",
    "PFE": "Pfizer Inc.",
    "KO": "Coca-Cola Co.",
    "WMT": "Walmart Inc.",
    "VZ": "Verizon Communications",
    "CMCSA": "Comcast Corp.",
    "CRM": "Salesforce.com Inc.",
    "INTC": "Intel Corp.",
    "T": "AT&T Inc.",
    "CSCO": "Cisco Systems Inc.",
    "CVX": "Chevron Corp.",
    "NKE": "Nike Inc.",
    "ORCL": "Oracle Corp.",
    "ABT": "Abbott Laboratories",
    "PEP": "PepsiCo Inc.",
    "MRK": "Merck & Co. Inc.",
    "MCD": "McDonald’s Corp.",
    "ADBE": "Adobe Inc.",
    "ACN": "Accenture PLC",
    "MDT": "Medtronic PLC",
    "TMO": "Thermo Fisher Scientific Inc.",
    "HON": "Honeywell International Inc.",
    "UNH": "UnitedHealth Group Inc.",
    "BA": "Boeing Co.",
    "ABBV": "AbbVie Inc.",
    "AMAT": "Applied Materials Inc.",
    "COST": "Costco Wholesale Corp.",
    "DHR": "Danaher Corp.",
    "LIN": "Linde PLC",
    "QCOM": "Qualcomm Inc.",
    "IBM": "IBM Corp.",
    "SBUX": "Starbucks Corp.",
    "GE": "General Electric Co.",
    "MMM": "3M Co.",
    "UPS": "United Parcel Service Inc.",
    "LMT": "Lockheed Martin Corp.",
    "CAT": "Caterpillar Inc.",
    "GS": "Goldman Sachs Group Inc.",
    "BLK": "BlackRock Inc.",
    "AMGN": "Amgen Inc.",
    "SPGI": "S&P Global Inc.",
    "TXN": "Texas Instruments Inc.",
    "C": "Citigroup Inc.",
    "ADP": "Automatic Data Processing Inc.",
    "USB": "U.S. Bancorp",
    "NOW": "ServiceNow Inc.",
    "MO": "Altria Group Inc.",
    "CI": "Cigna Corp.",
    "TGT": "Target Corp.",
    "SCHW": "Charles Schwab Corp.",
    "EQIX": "Equinix Inc.",
    "BKNG": "Booking Holdings Inc.",
    "ISRG": "Intuitive Surgical Inc.",
    "REGN": "Regeneron Pharmaceuticals Inc.",
    "BIIB": "Biogen Inc.",
    "FDX": "FedEx Corp.",
    "AXP": "American Express Co.",
    "DE": "Deere & Co.",
    "DD": "DuPont de Nemours Inc.",
    "EMR": "Emerson Electric Co.",
    "APD": "Air Products & Chemicals Inc.",
    "SHW": "Sherwin-Williams Co.",
    "FIS": "Fidelity National Information Services",
    "ADI": "Analog Devices Inc.",
    "STZ": "Constellation Brands Inc.",
    "ZTS": "Zoetis Inc.",
    "GILD": "Gilead Sciences Inc.",
    "ECL": "Ecolab Inc.",
    "OXY": "Occidental Petroleum Corp.",
    "SLB": "Schlumberger Ltd.",
    "WFC": "Wells Fargo & Co.",
    "PGR": "Progressive Corp.",
    "MET": "MetLife Inc.",
    "ICE": "Intercontinental Exchange Inc.",
    "ALL": "The Allstate Corp.",
    "PNC": "PNC Financial Services",
    "RTX": "Raytheon Technologies Corp.",
    "ANTM": "Anthem Inc.",
    "EW": "Edwards Lifesciences Corp.",
    "ITW": "Illinois Tool Works Inc.",
    "EOG": "EOG Resources Inc.",
    "LOW": "Lowe's Companies Inc.",
    "CTAS": "Cintas Corp.",
    "LRCX": "Lam Research Corp.",
    "SPG": "Simon Property Group Inc.",
    "GM": "General Motors Co.",
    "F": "Ford Motor Co.",
    "NSC": "Norfolk Southern Corp.",
    "PLD": "Prologis Inc.",
    "AMT": "American Tower Corp.",
    "CCI": "Crown Castle International Corp.",
    "DUK": "Duke Energy Corp.",
    "SO": "Southern Co.",
    "EXC": "Exelon Corp.",
    "AEP": "American Electric Power",
    "SRE": "Sempra Energy",
    "CMI": "Cummins Inc.",
    "HPQ": "HP Inc.",
    "BK": "The Bank of New York Mellon Corp.",
    "APTV": "Aptiv PLC",
    "VLO": "Valero Energy Corp.",
    "FTNT": "Fortinet Inc.",
    "CSX": "CSX Corp."
}

#############################################
# Helper Function: Fetch data with retries and caching
#############################################
@lru_cache(maxsize=32)
def fetch_data_cached(tickers, period, auto_adjust, column):
    # Convert tickers to a tuple for caching.
    tickers = tuple(tickers)
    for attempt in range(3):
        try:
            data = yf.download(list(tickers), period=period, auto_adjust=auto_adjust)[column]
            data = data.dropna()
            if data.empty:
                raise ValueError("Data is empty")
            return data
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {tickers}: {e}")
            time.sleep(2)
    return pd.DataFrame()

#############################################
# BudgetCSP Class (Discrete Optimization with Advanced Risk and Diversification Constraints)
#############################################
class BudgetCSP:
    def __init__(self, income, expenses, required_savings, risk_tolerance, curated_options=None, current_portfolio=None, step=50, min_stocks=2):
        self.income = income
        self.expenses = expenses
        self.required_savings = required_savings
        self.risk_tolerance = risk_tolerance.lower()
        self.current_portfolio = current_portfolio or {}
        self.step = step
        self.min_stocks = min_stocks  # Minimum number of stocks that must have a positive allocation
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
        base_max = int(self._available_budget * risk_pct)
        for stock in self.curated_options:
            step_max = (base_max // self.step) * self.step
            final_max = min(self.investment_options[stock]['max'], step_max)
            self.investment_options[stock]['max'] = final_max
            print(f"  {stock} - {TICKER_TO_NAME.get(stock, 'Unknown')}: ${final_max}")

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
        print("\nDefined Variables and Domains:")
        for stock, domain in self.variables.items():
            print(f"  {stock} - {TICKER_TO_NAME.get(stock, 'Unknown')}: {domain}")

    def define_constraints(self):
        # Basic budget constraint.
        def total_constraint(assignment):
            return sum(assignment.values()) <= self._available_budget
        
        # Advanced risk constraint using portfolio variance.
        # Relaxed thresholds: low: 0.10, medium: 0.15, high: 0.20 (annualized variance)
        risk_thresholds = {"low": 0.10, "medium": 0.15, "high": 0.20}
        def risk_constraint(assignment):
            total_alloc = sum(assignment.values())
            if total_alloc == 0:
                return True
            weights = []
            for stock in self.curated_options:
                alloc = assignment.get(stock, 0)
                weights.append(alloc / total_alloc)
            weights = np.array(weights)
            try:
                data = fetch_data_cached(self.curated_options, period="60d", auto_adjust=True, column='Close')
                if data.empty:
                    return True
                returns = data.pct_change().dropna()
                cov_matrix = returns.cov() * 252
                port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                return port_variance <= risk_thresholds[self.risk_tolerance]
            except Exception as e:
                return True

        # New diversification constraint: at least min_stocks must have > 0 allocation.
        def diversification_constraint(assignment):
            nonzero_count = sum(1 for alloc in assignment.values() if alloc > 0)
            return nonzero_count >= self.min_stocks

        self.constraints.append(total_constraint)
        self.constraints.append(risk_constraint)
        self.constraints.append(diversification_constraint)
        print("\nConstraints defined:")
        print(f"  - Total investment <= ${self._available_budget}")
        print(f"  - Portfolio variance <= {risk_thresholds[self.risk_tolerance]} (risk tolerance: {self.risk_tolerance})")
        print(f"  - At least {self.min_stocks} stocks must have nonzero allocation")

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

#############################################
# PortfolioOptimizer Class (Continuous Optimization)
#############################################
class PortfolioOptimizer:
    def __init__(self, tickers, available_budget, risk_free_rate=0.01, lookback_days=252):
        self.tickers = tickers
        self.available_budget = available_budget
        self.risk_free_rate = risk_free_rate
        self.lookback_days = lookback_days
        self.returns = None
        self.expected_returns = None
        self.cov_matrix = None
        self.weights = None

    def fetch_data(self):
        data = fetch_data_cached(tuple(self.tickers), period=f"{self.lookback_days}d", auto_adjust=False, column='Adj Close')
        return data

    def compute_statistics(self, prices):
        returns = prices.pct_change().dropna()
        self.returns = returns
        mu = returns.mean() * 252
        sigma = returns.cov() * 252
        self.expected_returns = mu
        self.cov_matrix = sigma

    def sharpe_ratio(self, weights):
        port_return = np.dot(weights, self.expected_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        if port_vol == 0:
            return -np.inf
        sharpe = (port_return - self.risk_free_rate) / port_vol
        return sharpe

    def negative_sharpe(self, weights):
        return -self.sharpe_ratio(weights)

    def optimize_portfolio(self):
        n = len(self.tickers)
        x0 = np.ones(n) / n
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(n))
        result = minimize(self.negative_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            self.weights = result.x
        else:
            raise ValueError("Optimization did not converge")
        return self.weights

    def get_allocations(self):
        if self.weights is None:
            raise ValueError("You must run optimize_portfolio() first.")
        allocations = {ticker: weight * self.available_budget for ticker, weight in zip(self.tickers, self.weights)}
        return allocations

    def rebalance(self):
        prices = self.fetch_data()
        self.compute_statistics(prices)
        self.optimize_portfolio()
        alloc = self.get_allocations()
        return alloc

#############################################
# New Test Cases for Both Modules
#############################################
def run_budget_csp_tests():
    print("\n====== BudgetCSP Test 1: Full Allocation ======")
    csp1 = BudgetCSP(
        income=8000,
        expenses=5000,
        required_savings=1000,
        risk_tolerance="high",
        curated_options=["TSLA", "NVDA", "AMD"],
        current_portfolio={"AAPL": 500},
        min_stocks=2
    )
    sol1 = csp1.solve()
    print("BudgetCSP Optimal Allocation:")
    print(sol1)
    if sol1:
        total = sum(sol1.values())
        print(f"Budget Utilization: {total}/{csp1._available_budget} ({total/csp1._available_budget:.1%})")

    print("\n====== BudgetCSP Test 2: Partial Allocation ======")
    csp2 = BudgetCSP(
        income=5000,
        expenses=4000,
        required_savings=800,
        risk_tolerance="medium",
        curated_options=["GOOGL", "AMZN"],
        min_stocks=1
    )
    sol2 = csp2.solve()
    print("BudgetCSP Best Solution:", sol2 or "No valid allocation")

def run_portfolio_optimizer_tests():
    print("\n====== PortfolioOptimizer Test 1: Standard Optimization ======")
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JNJ", "V", "PG"]
    available_budget = 10000
    optimizer = PortfolioOptimizer(tickers, available_budget, risk_free_rate=0.01)
    allocations = optimizer.rebalance()
    print("\nPortfolioOptimizer Optimal Dollar Allocations:")
    for ticker, amount in allocations.items():
        company_name = TICKER_TO_NAME.get(ticker, "Unknown")
        print(f"  {ticker} - {company_name}: ${amount:.2f}")
    print("\nPortfolioOptimizer Optimal Weights:")
    for ticker, weight in zip(tickers, optimizer.weights):
        print(f"  {ticker} - {TICKER_TO_NAME.get(ticker, 'Unknown')}: {weight:.2%}")
    sharpe = optimizer.sharpe_ratio(optimizer.weights)
    print(f"\nPortfolioOptimizer Sharpe Ratio: {sharpe:.2f}")

def run_extended_tests():
    print("\n=== BudgetCSP Test 4: Low Risk Allocation ===")
    csp4 = BudgetCSP(
        income=10000,
        expenses=6000,
        required_savings=2000,
        risk_tolerance="low",
        curated_options=["JNJ", "PG", "KO"],
        step=100,
        min_stocks=2
    )
    sol4 = csp4.solve()
    print("BudgetCSP Conservative Allocation:", sol4)
    if sol4:
        total = sum(sol4.values())
        print(f"Used ${total} of ${csp4._available_budget}")

    print("\n=== BudgetCSP Test 5: Small Step Size ===")
    csp5 = BudgetCSP(
        income=3000,
        expenses=2000,
        required_savings=500,
        risk_tolerance="medium",
        curated_options=["F", "GM"],
        step=10,
        min_stocks=1
    )
    sol5 = csp5.solve()
    print("BudgetCSP Precise Allocation:", sol5)
    
    print("\n=== PortfolioOptimizer Test 2: Rebalance with Extended Lookback ===")
    tickers2 = ["BRK-A", "AMZN", "F"]
    optimizer2 = PortfolioOptimizer(tickers2, available_budget=15000, risk_free_rate=0.01, lookback_days=365)
    allocations2 = optimizer2.rebalance()
    print("\nPortfolioOptimizer Extended Optimal Dollar Allocations:")
    for ticker, amount in allocations2.items():
        print(f"  {ticker} - {TICKER_TO_NAME.get(ticker, 'Unknown')}: ${amount:.2f}")
    print("\nPortfolioOptimizer Extended Optimal Weights:")
    for ticker, weight in zip(tickers2, optimizer2.weights):
        print(f"  {ticker} - {TICKER_TO_NAME.get(ticker, 'Unknown')}: {weight:.2%}")
    sharpe2 = optimizer2.sharpe_ratio(optimizer2.weights)
    print(f"\nPortfolioOptimizer Extended Sharpe Ratio: {sharpe2:.2f}")

if __name__ == "__main__":
    print("=== Running BudgetCSP Tests ===")
    run_budget_csp_tests()
    run_extended_tests()
    print("\n=== Running PortfolioOptimizer Tests ===")
    run_portfolio_optimizer_tests()
