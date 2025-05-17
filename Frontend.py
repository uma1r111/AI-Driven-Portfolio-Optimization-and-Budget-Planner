import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import time
import random
from PIL import Image
import base64
from io import BytesIO

# Import your optimizer modules - just reference imports for the frontend
# These would be properly connected in a real implementation
from csp_v3 import BudgetCSP, TICKER_TO_NAME, fetch_data_cached
from Astar_v2 import AStarPortfolioOptimizer, load_heuristics

# Define ticker-to-company name mapping (copied from your code)
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
    "MCD": "McDonald's Corp.",
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

# Styling and layout configurations
st.set_page_config(
    page_title="Smart Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Define the primary red color palette */
    :root {
        --primary-red: #E53935;
        --primary-red-dark: #C62828;
        --primary-red-light: #EF5350;
        --primary-red-pale: #FFCDD2;
    }

    .main-header {
        font-size: 2.5rem;
        color: #E53935;  /* Updated to red */
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .portfolio-header {
        background-color: #E53935;  /* Updated to red */
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        font-weight: 600;
        text-align: center;
        margin: 1rem 0;
    }
    
    .metric-container {
        display: flex;
        justify-content: space-between;
        background-color: #f9f9f9;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .high-sharpe {
        color: #388E3C;
        font-weight: 600;
    }
    
    .high-return {
        color: #E53935;  /* Updated to red */
        font-weight: 600;
    }
    
    .best-efficiency {
        color: #E53935;  /* Updated to red */
        font-weight: 600;
    }
    
    .recommendation-box {
        background-color: #E53935;  /* Updated to red */
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #B71C1C;  /* Darker red */
        margin-top: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .recommendation-box h3 {
        color: white;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .recommendation-box h4 {
        color: #FFEBEE;  /* Very light red */
        font-weight: 600;
        margin-top: 1rem;
    }
    
    .recommendation-box ul {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 1rem 1rem 1rem 2rem;
        border-radius: 0.3rem;
    }
    
    .recommendation-box li {
        margin-bottom: 0.5rem;
    }
    
    .recommendation-box p em {
        color: #FFEBEE;  /* Very light red */
        font-style: italic;
    }
    
    .loading-spinner {
        text-align: center;
        padding: 2rem;
    }
    
    /* Tabs styling - updated to red theme */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #FFEBEE;  /* Very light red background */
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #C62828;  /* Darker red text */
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #E53935;  /* Red background for active tab */
        color: white;
        font-weight: 600;
    }
    
    /* Button styling - updated to red theme */
    .stButton > button {
        background-color: #E53935;  /* Red background */
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #C62828;  /* Darker red on hover */
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #FFEBEE;  /* Very light red background */
    }
    
    /* Streamlit default element overrides */
    .st-emotion-cache-16idsys p {
        font-size: 14px;
    }
    
    /* Metric indicators */
    .st-emotion-cache-1wivap2 {
        color: #E53935;  /* Update primary metric color to red */
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<div class="main-header">📊 Smart Portfolio Optimizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Optimize your investment portfolio using advanced AI algorithms</div>', unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if "portfolios_generated" not in st.session_state:
    st.session_state.portfolios_generated = False

# Create sidebar with input form
with st.sidebar:
    st.header("📝 Investment Parameters")
    
    # Financial inputs
    st.subheader("Financial Information")
    user_income = st.number_input("Monthly Income ($)", min_value=0, value=12000, step=500)
    user_expenses = st.number_input("Monthly Expenses ($)", min_value=0, value=5000, step=500)
    user_savings = st.number_input("Required Savings ($)", min_value=0, value=2000, step=500)
    
    # Risk preference
    st.subheader("Risk Preference")
    user_risk = st.select_slider(
        "Risk Tolerance",
        options=["low", "medium", "high"],
        value="medium",
        help="Low: Conservative approach with lower returns. High: Aggressive approach with potentially higher returns."
    )
    
    # Stock selection
    st.subheader("Stock Selection")
    
    # List of available stocks
    available_stocks = list(TICKER_TO_NAME.keys())
    
    # Multi-select for stocks
    selected_stocks = st.multiselect(
        "Select Preferred Stocks",
        options=available_stocks,
        default=["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"],
        format_func=lambda x: f"{x} - {TICKER_TO_NAME.get(x, 'Unknown')}",
        help="Select at least 5 stocks for better diversification options"
    )
    
    # Advanced options (collapsible)
    with st.expander("Advanced Options"):
        min_stock_count = st.slider("Minimum Stocks in Portfolio", min_value=1, max_value=10, value=3)
        step_size = st.select_slider("Investment Step Size ($)", options=[10, 50, 100, 500], value=50)
    
    # Generate button - clicking this will reset the portfolios_generated flag to force recalculation
    generate_button = st.button("🚀 Generate Portfolios", type="primary", use_container_width=True)
    if generate_button:
        # When the button is clicked, reset the flag to trigger recalculation
        st.session_state.portfolios_generated = False
    
    # Add disclaimer
    st.caption("This tool provides educational guidance only. Always consult with a financial advisor before making investment decisions.")

# Main area
# Now check if we should run optimization
if generate_button or st.session_state.portfolios_generated:
    # Validation checks
    if len(selected_stocks) < min_stock_count:
        st.error(f"Please select at least {min_stock_count} stocks to meet your minimum portfolio diversity requirement.")
    elif user_income <= user_expenses + user_savings:
        st.error("Your income must be greater than the sum of expenses and required savings.")
    else:
        # Set the flag if not already set
        if not st.session_state.portfolios_generated:
            st.session_state.portfolios_generated = True
            
            # Show loading spinner while "calculating"
            with st.spinner("Optimizing portfolios... This may take a moment."):
                time.sleep(3)  # Simulate calculation time
                
                # In a real app, this is where you'd call your backend logic:
                # 1. Generate stock combinations
                # 2. Run your optimizer for each combination
                # 3. Calculate metrics and store results
                
                # For demo purposes, we'll create synthetic portfolio data
                # Normally this would come from your actual optimizer functions
                
                # Create 3 random portfolio combinations
                random.shuffle(selected_stocks)
                stock_combinations = [
                    selected_stocks[:min(len(selected_stocks), min_stock_count+2)],
                    selected_stocks[1:min(len(selected_stocks), min_stock_count+3)],
                    selected_stocks[2:min(len(selected_stocks), min_stock_count+2)]
                ]
                
                # Create synthetic portfolio data for the UI demonstration
                def generate_mock_portfolio(stocks, available_budget):
                    portfolio = {}
                    total_allocation = 0
                    
                    # Randomly allocate to stocks, ensuring we meet min_stock_count
                    while sum(1 for v in portfolio.values() if v > 0) < min_stock_count:
                        for stock in stocks:
                            # Random allocation in steps
                            steps = random.randint(0, int(available_budget/(len(stocks)*step_size)))
                            allocation = steps * step_size
                            
                            # Ensure we don't exceed budget
                            if total_allocation + allocation <= available_budget:
                                portfolio[stock] = allocation
                                total_allocation += allocation
                            
                            # Stop if we've allocated enough
                            if total_allocation >= available_budget * 0.85:
                                break
                    
                    # Fill in any missing stocks with zero allocation
                    for stock in stocks:
                        if stock not in portfolio:
                            portfolio[stock] = 0
                    
                    return portfolio
                
                # Calculate available budget
                available_budget = user_income - user_expenses - user_savings
                
                # Generate mock portfolios
                portfolios = []
                summary_data = []
                
                for i, stocks in enumerate(stock_combinations):
                    # Generate the portfolio
                    portfolio = generate_mock_portfolio(stocks, available_budget)
                    
                    # Calculate metrics (in a real app these would come from your optimizer)
                    total_investment = sum(portfolio.values())
                    
                    # Create mock heuristics
                    heuristics = {stock: random.uniform(0.5, 1.5) for stock in stocks}
                    total_score = sum(heuristics.get(stock, 0) * amount for stock, amount in portfolio.items())
                    avg_score_per_dollar = total_score / total_investment if total_investment > 0 else 0
                    sharpe_ratio = random.uniform(0.8, 1.5)
                    expected_return = random.uniform(0.05, 0.15)
                    volatility = random.uniform(0.08, 0.20)
                    
                    # Create portfolio detail dataframe
                    portfolio_df = pd.DataFrame([
                        {
                            "Stock": stock,
                            "Company Name": TICKER_TO_NAME.get(stock, "Unknown"),
                            "Allocation ($)": amount,
                            "Heuristic Score": round(heuristics.get(stock, 0), 4),
                            "% of Total": round(100 * amount / total_investment if total_investment > 0 else 0, 2)
                        }
                        for stock, amount in portfolio.items() if amount > 0
                    ])
                    
                    # Sort by allocation
                    portfolio_df = portfolio_df.sort_values(by="Allocation ($)", ascending=False)
                    
                    # Store portfolio data
                    portfolios.append({
                        "id": i+1,
                        "stocks": stocks,
                        "portfolio": portfolio,
                        "dataframe": portfolio_df,
                        "total_investment": total_investment,
                        "total_score": total_score,
                        "avg_score_per_dollar": avg_score_per_dollar,
                        "sharpe_ratio": sharpe_ratio,
                        "expected_return": expected_return,
                        "volatility": volatility
                    })
                    
                    # Add to summary data
                    summary_data.append({
                        "Portfolio": i+1,
                        "Stocks Used": len(portfolio_df),
                        "Total Investment": total_investment,
                        "Total Score": round(total_score, 2),
                        "Avg Score/$": round(avg_score_per_dollar, 4),
                        "Sharpe Ratio": round(sharpe_ratio, 3),
                        "Return (%)": round(expected_return * 100, 2),
                        "Volatility (%)": round(volatility * 100, 2),
                        "Included Stocks": ", ".join([s for s, v in portfolio.items() if v > 0])
                    })
                
                # Store in session state
                st.session_state.portfolios = portfolios
                st.session_state.summary_data = summary_data
                
                # Determine best portfolios by different metrics
                best_sharpe_idx = max(range(len(portfolios)), key=lambda i: portfolios[i]["sharpe_ratio"])
                best_return_idx = max(range(len(portfolios)), key=lambda i: portfolios[i]["expected_return"])
                best_efficiency_idx = max(range(len(portfolios)), key=lambda i: portfolios[i]["avg_score_per_dollar"])
                
                # Create recommendation based on best overall (using a weighted score)
                weighted_scores = [(i, 
                                   p["sharpe_ratio"] * 0.4 + 
                                   p["expected_return"] * 0.3 + 
                                   p["avg_score_per_dollar"] * 0.3) 
                                  for i, p in enumerate(portfolios)]
                best_overall_idx = max(weighted_scores, key=lambda x: x[1])[0]
                
                st.session_state.best = {
                    "sharpe": best_sharpe_idx,
                    "return": best_return_idx,
                    "efficiency": best_efficiency_idx,
                    "overall": best_overall_idx
                }
        
        # Display results (either newly calculated or from session state)
        if st.session_state.portfolios_generated and hasattr(st.session_state, 'portfolios'):
            portfolios = st.session_state.portfolios
            summary_data = st.session_state.summary_data
            best = st.session_state.best
            
            # Display portfolio tabs
            tabs = st.tabs([f"Portfolio {p['id']}" for p in portfolios])
            
            # Populate each tab
            for i, tab in enumerate(tabs):
                p = portfolios[i]
                
                with tab:
                    # Portfolio header with special indicators
                    header_text = f"Portfolio {p['id']}"
                    if i == best["sharpe"]:
                        header_text += " 🛡️ (Best Stability)"
                    if i == best["return"]:
                        header_text += " 📈 (Best Growth)"
                    if i == best["efficiency"]:
                        header_text += " ⚡ (Most Efficient)"
                    
                    st.markdown(f"<div class='portfolio-header'>{header_text}</div>", unsafe_allow_html=True)
                    
                    # Key metrics row
                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        st.metric("Total Investment", f"${p['total_investment']:,.2f}")
                    with metric_cols[1]:
                        sharpe_display = p['sharpe_ratio']
                        if i == best["sharpe"]:
                            st.metric("Sharpe Ratio", f"{sharpe_display:.3f} 🏆")
                        else:
                            st.metric("Sharpe Ratio", f"{sharpe_display:.3f}")
                    with metric_cols[2]:
                        return_display = p['expected_return'] * 100
                        if i == best["return"]:
                            st.metric("Expected Return", f"{return_display:.2f}% 🏆")
                        else:
                            st.metric("Expected Return", f"{return_display:.2f}%")
                    with metric_cols[3]:
                        st.metric("Volatility", f"{p['volatility']*100:.2f}%")
                    
                    # Portfolio allocation table
                    st.subheader("Stock Allocations")
                    st.dataframe(
                        p["dataframe"],
                        hide_index=True,
                        column_config={
                            "Stock": st.column_config.TextColumn("Stock"),
                            "Company Name": st.column_config.TextColumn("Company Name"),
                            "Allocation ($)": st.column_config.NumberColumn(
                                "Allocation ($)",
                                format="$%.2f"
                            ),
                            "Heuristic Score": st.column_config.NumberColumn(
                                "Heuristic Score",
                                format="%.4f"
                            ),
                            "% of Total": st.column_config.ProgressColumn(
                                "% of Total",
                                format="%.2f%%",
                                min_value=0,
                                max_value=100
                            )
                        }
                    )
                    
                    # Create pie chart for this portfolio
                    pie_data = p["dataframe"]
                    if not pie_data.empty:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        
                        # Create custom color palette that matches the red theme
                        colors = ['#E53935', '#F44336', '#EF5350', '#E57373', '#EF9A9A', '#FFCDD2', 
                                '#B71C1C', '#C62828', '#D32F2F', '#E53935', '#F44336', '#EF5350']
                        
                        # Make sure we have enough colors (repeat if necessary)
                        while len(colors) < len(pie_data):
                            colors = colors + colors
                        
                        # Create explode effect to highlight largest allocation
                        max_idx = pie_data["Allocation ($)"].idxmax()
                        explode = [0.05 if i == max_idx else 0 for i in range(len(pie_data))]
                        
                        # Create the pie chart with improved styling
                        labels = pie_data["Stock"].tolist()
                        sizes = pie_data["Allocation ($)"].tolist()
                        
                        wedges, texts, autotexts = ax.pie(
                            sizes, 
                            labels=labels, 
                            autopct='%1.1f%%', 
                            startangle=90, 
                            shadow=False,
                            explode=explode,
                            colors=colors[:len(pie_data)],
                            wedgeprops=dict(width=0.5, edgecolor='w'),
                            textprops={'fontsize': 10, 'color': 'black', 'weight': 'bold'}
                        )
                        
                        # Make percentage text more readable
                        for autotext in autotexts:
                            autotext.set_color('white')
                            autotext.set_fontsize(9)
                            autotext.set_weight('bold')
                        
                        # Create the circle in the middle for a donut chart effect
                        centre_circle = plt.Circle((0, 0), 0.25, fc='white')
                        fig.gca().add_artist(centre_circle)
                        
                        # Set title with matching style
                        ax.set_title("Portfolio Allocation", fontsize=16, color='#C62828', fontweight='bold')
                        ax.axis('equal')
                        
                        # Add a legend below the chart with company names
                        legend_labels = [f"{stock} - {TICKER_TO_NAME.get(stock, 'Unknown')}" for stock in labels]
                        ax.legend(wedges, legend_labels, title="Stock Details", 
                                loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                        
                        # Improve overall appearance with red-tinted background
                        fig.patch.set_facecolor('#FFEBEE')  # Very light red background
                        
                        # Display the chart
                        st.pyplot(fig)
            
            # Display summary table
            st.subheader("Portfolio Comparison Summary")
            summary_df = pd.DataFrame(summary_data)
            
            # Format summary table with indicators for best portfolios
            def highlight_best(row):
                styles = [''] * len(row)
                portfolio_idx = int(row["Portfolio"]) - 1
                
                if portfolio_idx == best["sharpe"]:
                    styles[5] = 'background-color: #c8e6c9; font-weight: bold;'  # Light green
                if portfolio_idx == best["return"]:
                    styles[6] = 'background-color: #FFB6C1; font-weight: bold;'  # Light red
                if portfolio_idx == best["efficiency"]:
                    styles[4] = 'background-color: #bbdefb; font-weight: bold;'  # Light blue
                if portfolio_idx == best["overall"]:
                    styles[0] = 'background-color: #d1c4e9; font-weight: bold;'  # Light purple
                
                return styles
            
            # Convert numeric columns for better formatting
            format_cols = {
                "Total Investment": "${:,.2f}",
                "Total Score": "{:,.2f}",
                "Sharpe Ratio": "{:.3f}",
                "Return (%)": "{:.2f}%",
                "Volatility (%)": "{:.2f}%"
            }
            
            for col, fmt in format_cols.items():
                summary_df[col] = summary_df[col].apply(lambda x: fmt.format(x))
            
            # Display styled dataframe
            st.dataframe(
                summary_df.style.apply(highlight_best, axis=1),
                hide_index=True,
                use_container_width=True
            )
            
            # Recommendation box
            best_portfolio = portfolios[best["overall"]]
            st.markdown(f"""
            <div class="recommendation-box">
                <h3>💡 Portfolio Recommendation</h3>
                <p><strong>Portfolio {best["overall"] + 1}</strong> is the recommended choice based on weighted analysis of return potential, stability, and efficiency.</p>
                <br>
                <h4>📌 Investment Guidance:</h4>
                <ul>
                    <li>If you're a <strong>conservative investor</strong>, prioritize Portfolio {best["sharpe"] + 1} for its higher Sharpe Ratio (better risk-adjusted returns).</li>
                    <li>If you want <strong>maximum growth potential</strong>, consider Portfolio {best["return"] + 1} for its higher expected returns.</li>
                    <li>If you prefer <strong>investment efficiency</strong>, look at Portfolio {best["efficiency"] + 1} which offers the best score per dollar invested.</li>
                </ul>
                <br>
                <p><em>Remember: Past performance is not indicative of future results. Diversification is key to reducing overall portfolio risk.</em></p>
            </div>
            """, unsafe_allow_html=True)
else:
    # Show welcome message and instructions if no portfolios have been generated
   
    # Display welcome message and instructions when no portfolios have been generated
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Example portfolio image or icon
        st.image("https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?q=80&w=1170&auto=format&fit=crop", 
                 use_container_width=True)
    
    with col2:
        st.markdown("## Welcome to Smart Portfolio Optimizer")
        st.markdown("""
        This tool helps you create optimized investment portfolios based on your financial situation and preferences.
        
        ### How to use:
        1. Fill in your financial information in the sidebar
        2. Select your risk tolerance level
        3. Choose stocks you're interested in (at least 5 recommended)
        4. Click "Generate Portfolios" to see optimized recommendations
        
        Our AI algorithms will create multiple portfolio options optimized for different objectives: stability, growth, and efficiency.
        """)
        
        st.info("📌 To get started, configure your parameters in the sidebar and click 'Generate Portfolios'")

# Add a reset button in the sidebar to clear session state and start over
with st.sidebar:
    st.markdown("---")
    if st.button("🔄 Reset All Parameters", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Fix to ensure portfolios are regenerated when parameters change
# Add these important parameters to check for changes
if "prev_params" not in st.session_state:
    st.session_state.prev_params = {
        "income": user_income,
        "expenses": user_expenses,
        "savings": user_savings,
        "risk": user_risk,
        "stocks": selected_stocks,
        "min_stock_count": min_stock_count,
        "step_size": step_size
    }
    
# Check if any parameters have changed
current_params = {
    "income": user_income,
    "expenses": user_expenses,
    "savings": user_savings,
    "risk": user_risk,
    "stocks": selected_stocks,
    "min_stock_count": min_stock_count,
    "step_size": step_size
}

# If parameters changed, reset the portfolios_generated flag

# Update the previous parameters
st.session_state.prev_params = current_params.copy()

