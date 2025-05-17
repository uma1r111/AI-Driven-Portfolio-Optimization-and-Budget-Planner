import pandas as pd
import numpy as np

df = pd.read_csv('/combined_stock_features.csv')
print(df)

# Create a mapping for Ticker and Sector
ticker_to_id = {ticker: idx for idx, ticker in enumerate(df['Ticker'].unique())}
sector_to_id = {sector: idx for idx, sector in enumerate(df['Sector'].unique())}

# Add Ticker_ID and Sector_ID columns
df['Ticker_ID'] = df['Ticker'].map(ticker_to_id)
df['Sector_ID'] = df['Sector'].map(sector_to_id)

ticker_to_id

sector_to_id

# Select features and target
features = ['Close', 'MA100', 'RSI', 'Norm_Volume', 'Bollinger_Upper', 'Bollinger_Lower', 'Log_Returns', 'Ticker_ID', 'Sector_ID']
data = df[features]  # Keep only selected features

data['Sector_ID'].unique()

def min_max_scaling(data):
    """Scale data to [0, 1] range and return scaled data + min/max values"""
    min_val = np.min(data)
    max_val = np.max(data)
    # Handle case where all values are identical (avoid division by zero)
    if max_val == min_val:
        return np.zeros_like(data), min_val, max_val
    return (data - min_val) / (max_val - min_val), min_val, max_val

def scale_features(df):
    scaled_data = {}
    scalers = {}

    for col in df.columns:
        if col in ['Ticker_ID', 'Sector_ID']:  # Don't scale these columns
            scaled_data[col] = df[col].values
            continue

        scaled_values, min_val, max_val = min_max_scaling(df[col].values)
        if col == 'Log_Returns':
            scalers['target'] = (min_val, max_val)
        else:
            scalers[col] = (min_val, max_val)

        scaled_data[col] = scaled_values

    return pd.DataFrame(scaled_data), scalers

# Scale all features
scaled_data, scalers = scale_features(data)

# Split into train/test (keeping temporal order)
split_ratio = 0.8
split_index = int(len(scaled_data) * split_ratio)
train_data = scaled_data.iloc[:split_index]
test_data = scaled_data.iloc[split_index:]

print(train_data.columns)
print(test_data.columns)

# Create ticker mapping and ticker IDs
unique_tickers = list(set(train_data['Ticker_ID'].values) | set(test_data['Ticker_ID'].values))
ticker_mapping = {ticker: idx for idx, ticker in enumerate(unique_tickers)}

unique_sectors = list(set(train_data['Sector_ID'].values) | set(test_data['Sector_ID'].values))
sector_mapping = {sector_id: sector_id for sector_id in unique_sectors}

train_ticker_ids = train_data['Ticker_ID'].values
test_ticker_ids = test_data['Ticker_ID'].values

# Create sequences function for sector-wise training
def create_sequences_multi(data, ticker_ids, sector_ids, seq_length=50, target_col='Log_Returns'):
    """Create sequences from multiple features (without including Ticker_ID) for sector-wise training"""
    X, y, ticker_seq_list, sector_seq_list = [], [], [], []
    data_values = data.drop(['Ticker_ID', 'Sector_ID'], axis=1).values  # Drop 'Ticker_ID' and 'Sector_ID' from features

    for sector_id in np.unique(sector_ids):
        sector_data = data[data['Sector_ID'] == sector_id]
        sector_ticker_ids = sector_data['Ticker_ID'].values

        for ticker_id in np.unique(sector_ticker_ids):
            ticker_data = sector_data[sector_data['Ticker_ID'] == ticker_id]
            features = ticker_data.drop(['Log_Returns', 'Ticker_ID', 'Sector_ID'], axis=1).values
            targets = ticker_data['Log_Returns'].values

            for i in range(len(features) - seq_length):
                X.append(features[i:i+seq_length, :])  # Selecting only the features
                ticker_seq_list.append(ticker_id)
                sector_seq_list.append(sector_id)  # Include the sector_id in the sequence
                y.append(targets[i + seq_length])

    return np.array(X), np.array(y), np.array(ticker_seq_list), np.array(sector_seq_list)


# Create sequences for train and test data sector-wise
seq_length=50
X_train, y_train, train_tickers, train_sectors = create_sequences_multi(train_data, train_ticker_ids, train_data['Sector_ID'].values, seq_length=50)
X_test, y_test, test_tickers, test_sectors = create_sequences_multi(test_data, test_ticker_ids, test_data['Sector_ID'].values, seq_length=50)

# Check shapes
print(f"Training shapes - X: {X_train.shape}, y: {y_train.shape}")
print(f"Test shapes - X: {X_test.shape}, y: {y_test.shape}")

class LSTM:
    def __init__(self, input_dim, hidden_dim, output_dim, ticker_dim, embedding_dim, sector_dim, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = learning_rate
        self.ticker_dim = ticker_dim
        self.embedding_dim = embedding_dim
        self.sector_dim = sector_dim  # Number of sectors
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 1  # Time step for Adam

        # Combined input size: hidden + input + ticker + sector embedding
        concat_dim = hidden_dim + input_dim + embedding_dim + sector_dim

        # Initialize weights
        self.Wf = np.random.randn(hidden_dim, concat_dim) * 0.01
        self.bf = np.zeros((hidden_dim, 1))

        self.Wi = np.random.randn(hidden_dim, concat_dim) * 0.01
        self.bi = np.zeros((hidden_dim, 1))

        self.Wc = np.random.randn(hidden_dim, concat_dim) * 0.01
        self.bc = np.zeros((hidden_dim, 1))

        self.Wo = np.random.randn(hidden_dim, concat_dim) * 0.01
        self.bo = np.zeros((hidden_dim, 1))

        # Output layer
        self.Wy = np.random.randn(output_dim, hidden_dim) * 0.01
        self.by = np.zeros((output_dim, 1))

        # Initialize embedding matrices
        self.ticker_embedding = np.random.randn(self.ticker_dim, self.embedding_dim) * 0.01
        self.sector_embedding = np.random.randn(self.sector_dim, self.embedding_dim) * 0.01

        # Initialize Adam moment estimates
        self._init_adam_params()

    def _init_adam_params(self):
        self.m = {}
        self.v = {}
        for param_name in ['Wf', 'Wi', 'Wc', 'Wo', 'Wy', 'bf', 'bi', 'bc', 'bo', 'by']:
            param = getattr(self, param_name)
            self.m[param_name] = np.zeros_like(param)
            self.v[param_name] = np.zeros_like(param)

    def get_ticker_embedding(self, ticker_id):
        """Return the embedding for the ticker ID"""
        return self.ticker_embedding[ticker_id].reshape(-1, 1)

    def get_sector_embedding(self, sector_id):
        """Return the embedding for the sector ID"""
        return self.sector_embedding[sector_id].reshape(-1, 1)

    def sigmoid(self, x): return 1 / (1 + np.exp(-x))
    def dsigmoid(self, x): return x * (1 - x)
    def tanh(self, x): return np.tanh(x)
    def dtanh(self, x): return 1 - x ** 2

    def forward(self, x_seq, ticker_id, sector_id, h=None, c=None):
        if h is None:
            h = np.zeros((self.hidden_dim, 1))
        if c is None:
            c = np.zeros((self.hidden_dim, 1))
        self.caches = []

        # Get the ticker and sector embeddings once per sequence (not per timestep)
        ticker_embedding = self.get_ticker_embedding(ticker_id).reshape(-1, 1)
        sector_embedding = self.get_sector_embedding(sector_id).reshape(-1, 1)

        for x in x_seq:
            x = x.reshape(self.input_dim, 1)

            # Concatenate previous hidden state, input, ticker embedding, and sector embedding
            concat = np.vstack((h, x, ticker_embedding, sector_embedding))

            ft = self.sigmoid(np.dot(self.Wf, concat) + self.bf)
            it = self.sigmoid(np.dot(self.Wi, concat) + self.bi)
            c_tilde = self.tanh(np.dot(self.Wc, concat) + self.bc)
            c = ft * c + it * c_tilde
            ot = self.sigmoid(np.dot(self.Wo, concat) + self.bo)
            h = ot * self.tanh(c)

            self.caches.append((h, c, ft, it, c_tilde, ot, concat))

        y_hat = np.dot(self.Wy, h) + self.by
        return y_hat, h, c

    def backward(self, x_seq, y_hat, y_true):
        dh_next = np.zeros((self.hidden_dim, 1))
        dc_next = np.zeros((self.hidden_dim, 1))

        grads = {
            'Wf': np.zeros_like(self.Wf), 'Wi': np.zeros_like(self.Wi),
            'Wc': np.zeros_like(self.Wc), 'Wo': np.zeros_like(self.Wo),
            'Wy': np.zeros_like(self.Wy),
            'bf': np.zeros_like(self.bf), 'bi': np.zeros_like(self.bi),
            'bc': np.zeros_like(self.bc), 'bo': np.zeros_like(self.bo),
            'by': np.zeros_like(self.by)
        }

        dy = y_hat - y_true
        grads['Wy'] += np.dot(dy, self.caches[-1][0].T)
        grads['by'] += dy

        dh = np.dot(self.Wy.T, dy) + dh_next

        for t in reversed(range(len(x_seq))):
            h, c, ft, it, c_tilde, ot, concat = self.caches[t]
            c_prev = self.caches[t - 1][1] if t > 0 else np.zeros_like(c)

            do = dh * self.tanh(c)
            do_raw = do * self.dsigmoid(ot)

            dc = dh * ot * self.dtanh(self.tanh(c)) + dc_next
            dc_tilde = dc * it
            dc_tilde_raw = dc_tilde * self.dtanh(c_tilde)

            di = dc * c_tilde
            di_raw = di * self.dsigmoid(it)

            df = dc * c_prev
            df_raw = df * self.dsigmoid(ft)

            grads['Wf'] += np.dot(df_raw, concat.T)
            grads['Wi'] += np.dot(di_raw, concat.T)
            grads['Wc'] += np.dot(dc_tilde_raw, concat.T)
            grads['Wo'] += np.dot(do_raw, concat.T)

            grads['bf'] += df_raw
            grads['bi'] += di_raw
            grads['bc'] += dc_tilde_raw
            grads['bo'] += do_raw

            dconcat = (np.dot(self.Wf.T, df_raw) +
                       np.dot(self.Wi.T, di_raw) +
                       np.dot(self.Wc.T, dc_tilde_raw) +
                       np.dot(self.Wo.T, do_raw))

            dh = dconcat[:self.hidden_dim, :]
            dc_next = dc * ft

        self._apply_adam(grads)
        self.t += 1  # Increment timestep

    def _apply_adam(self, grads):
        for param_name in grads:
            grad = grads[param_name]
            self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
            self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)

            param = getattr(self, param_name)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            setattr(self, param_name, param)


    def train(self, X_train, y_train, ticker_ids_train, sector_ids_train, epochs=10, batch_size=32):
        for epoch in range(epochs):
            total_loss = 0
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)

            for i in range(0, len(X_train), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_loss = 0
                batch_grads = []

                for j in batch_indices:
                    x_seq = X_train[j]
                    y_true = y_train[j].reshape(self.output_dim, 1)
                    ticker_ids = ticker_ids_train[j]
                    sector_ids = sector_ids_train[j]
                    y_hat, _, _ = self.forward(x_seq, ticker_ids, sector_ids)
                    loss = np.mean((y_hat - y_true) ** 2)
                    batch_loss += loss

                    self.backward(x_seq, y_hat, y_true)

                total_loss += batch_loss

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.6f}")

# Print the sector to ID mapping
print(sector_to_id)

sector_mapping = {
    0: "Technology",
    1: "Healthcare",
    2: "Financial Services",
    3: "Physical Assets & Resources",
    4: "Consumer Cyclical",
    5: "Industrials",
    6: "Communication Services",
    7: "Consumer Defensive"
}

def train_lstm_sectorwise(X_train, y_train, train_tickers, train_sectors, sector_id, params, sector_models=None):
    if sector_models is None:
        sector_models = {}

    # Select data for the given sector
    sector_indices = np.where(train_sectors == sector_id)[0]

    if len(sector_indices) == 0:
        print(f"No training data for sector {sector_id}. Skipping.")
        return sector_models

    X_sector = X_train[sector_indices]
    y_sector = y_train[sector_indices]
    ticker_sector = train_tickers[sector_indices]
    sector_sector = train_sectors[sector_indices]

    # Get sector name for printing
    sector_name = sector_mapping.get(sector_id, f"Sector {sector_id}")

    # Check if model for sector already exists (for fine-tuning)
    if sector_id in sector_models:
        model = sector_models[sector_id]
        print(f"\nFine-tuning sector {sector_name}...")
    else:
        model = LSTM(
            input_dim=params['input_dim'],
            hidden_dim=params['hidden_dim'],
            output_dim=params['output_dim'],
            ticker_dim=params['ticker_dim'],
            embedding_dim=params['embedding_dim'],
            sector_dim=params['sector_dim'],
            learning_rate=params['learning_rate']
        )
        print(f"\nTraining {sector_name} sector from scratch...")

    # Train
    model.train(X_sector, y_sector, ticker_sector, sector_sector, epochs=params['epochs'], batch_size=params['batch_size'])

    # Save the model
    sector_models[sector_id] = model

    return sector_models

# Initialize empty models
sector_models = {}


#-x-x-x-x-x-x-x-x-x-x-x-x- TECHNOLOGY -x-x-x-x-x-x-x-x-x-x-x-x-

# Reverse mapping properly
id_to_ticker = {v: k for k, v in ticker_to_id.items()}  # use ticker_to_id

# Filter tickers belonging to Technology sector (sector_id = 0)
technology_sector_tickers = [ticker_id for ticker_id, sector_id in zip(train_tickers, train_sectors) if sector_id == 0]

# Get stock names correctly
technology_stock_names = [id_to_ticker[int(ticker_id)] for ticker_id in set(technology_sector_tickers)]

print("Stocks being trained in Technology sector:")
print(technology_stock_names)

params_technology = {
    'input_dim': X_train.shape[2],  # Number of features
    'hidden_dim': 64,
    'output_dim': 1,
    'ticker_dim': len(ticker_mapping),  # Total unique tickers
    'embedding_dim': 8,
    'sector_dim': len(sector_mapping),  # Total unique sectors
    'learning_rate': 0.001,
    'epochs': 6,
    'batch_size': 32
}

# Technology (sector_id -> 0)
sector_models = train_lstm_sectorwise(X_train, y_train, train_tickers, train_sectors, sector_id=0, params=params_technology, sector_models=sector_models)

def predict_logreturns(sector_model, X_test, test_tickers, sector_id, scalers):
    """
    Predict log returns for all stocks in the given sector.
    """
    predictions = []

    for i in range(len(X_test)):
        x_seq = X_test[i]
        ticker_id = test_tickers[i]

        # Reinitialize hidden and cell states
        h_prev = np.zeros((sector_model.hidden_dim, 1))
        c_prev = np.zeros((sector_model.hidden_dim, 1))

        # Get the prediction for this sequence
        y_pred, _, _ = sector_model.forward(x_seq, ticker_id, sector_id, h_prev, c_prev)
        predictions.append(y_pred.flatten()[0])

    # Inverse scale the predictions (log returns)
    min_target, max_target = scalers['target']
    predictions = np.array(predictions) * (max_target - min_target) + min_target

    return predictions

def calculate_volatility(log_returns, window_size=10):
    """
    Calculate volatility (standard deviation) of log returns for a given window size.
    """
    volatility = []
    for i in range(len(log_returns)):
        start = max(0, i - window_size + 1)
        window = log_returns[start:i+1]
        volatility.append(np.std(window))  # Standard deviation as volatility
    return np.array(volatility)

import os
def save_predictions_to_csv(sector_id, log_returns, volatility, test_tickers, output_filename, ticker_mapping, sector_mapping):
    """
    Save log returns and volatility to a CSV file for a specific sector with stock and sector names.
    """

    id_to_ticker = {v: k for k, v in ticker_to_id.items()}

    # Map ticker_ids to actual ticker names using ticker_mapping
    ticker_names = [id_to_ticker[ticker_id] for ticker_id in test_tickers]

    # Map sector_id to sector name using sector_mapping
    sector_name = sector_mapping.get(sector_id, "Unknown Sector")

    # Create DataFrame to store predictions
    df = pd.DataFrame({
        'Stock': ticker_names,
        'Log_Returns': log_returns,
        'Volatility': volatility,
        'Sector': [sector_name] * len(test_tickers)  # Add the sector name instead of sector_id
    })

    # Check if the file already exists to determine if headers should be written
    file_exists = os.path.exists(output_filename)

    # Append to the CSV file (with headers only if the file doesn't exist)
    df.to_csv(output_filename, index=False, mode='a', header=not file_exists)  # Only write header if the file doesn't exist

    print(f"Results for sector '{sector_name}' saved to {output_filename}")

technology_indices = [i for i, sector in enumerate(train_sectors) if sector == 0]

X_technology = [X_train[i] for i in technology_indices]
y_technology = [y_train[i] for i in technology_indices]
tickers_technology = [train_tickers[i] for i in technology_indices]

# Predict on same technology data
log_returns_technology = predict_logreturns(sector_models[0], X_technology, tickers_technology, sector_id=0, scalers=scalers)

# Calculate volatility
volatility_technology = calculate_volatility(log_returns_technology)

# Save to CSV
output_filename = 'technology_sector_predictions.csv'
save_predictions_to_csv(sector_id=0, log_returns=log_returns_technology, volatility=volatility_technology, test_tickers=tickers_technology, output_filename=output_filename, ticker_mapping=ticker_mapping, sector_mapping=sector_mapping)



# #-x-x-x-x-x-x-x-x-x-x-x-x- HEALTHCARE -x-x-x-x-x-x-x-x-x-x-x-x-

# # Reverse mapping properly
# id_to_ticker = {v: k for k, v in ticker_to_id.items()}  # use ticker_to_id

# # Filter tickers belonging to Health sector
# health_sector_tickers = [ticker_id for ticker_id, sector_id in zip(train_tickers, train_sectors) if sector_id == 1]

# # Get stock names correctly
# health_stock_names = [id_to_ticker[int(ticker_id)] for ticker_id in set(health_sector_tickers)]

# print("Stocks being trained in health sector:")
# print(health_stock_names)

# params_healthcare = {
#     'input_dim': X_train.shape[2],  # Number of features
#     'hidden_dim': 64,
#     'output_dim': 1,
#     'ticker_dim': len(ticker_mapping),  # Total unique tickers
#     'embedding_dim': 8,
#     'sector_dim': len(sector_mapping),  # Total unique sectors
#     'learning_rate': 0.001,
#     'epochs': 10,
#     'batch_size': 32
# }

# # Healthcare (sector_id -> 1)
# sector_models = train_lstm_sectorwise(X_train, y_train, train_tickers, train_sectors, sector_id=1, params=params_healthcare, sector_models=sector_models)

# def predict_logreturns(sector_model, X_test, test_tickers, sector_id, scalers):
#     """
#     Predict log returns for all stocks in the given sector.
#     """
#     predictions = []

#     for i in range(len(X_test)):
#         x_seq = X_test[i]
#         ticker_id = test_tickers[i]

#         # Reinitialize hidden and cell states
#         h_prev = np.zeros((sector_model.hidden_dim, 1))
#         c_prev = np.zeros((sector_model.hidden_dim, 1))

#         # Get the prediction for this sequence
#         y_pred, _, _ = sector_model.forward(x_seq, ticker_id, sector_id, h_prev, c_prev)
#         predictions.append(y_pred.flatten()[0])

#     # Inverse scale the predictions (log returns)
#     min_target, max_target = scalers['target']
#     predictions = np.array(predictions) * (max_target - min_target) + min_target

#     return predictions

# def calculate_volatility(log_returns, window_size=10):
#     """
#     Calculate volatility (standard deviation) of log returns for a given window size.
#     """
#     volatility = []
#     for i in range(len(log_returns)):
#         start = max(0, i - window_size + 1)
#         window = log_returns[start:i+1]
#         volatility.append(np.std(window))  # Standard deviation as volatility
#     return np.array(volatility)

# import os
# def save_predictions_to_csv(sector_id, log_returns, volatility, test_tickers, output_filename, ticker_mapping, sector_mapping):
#     """
#     Save log returns and volatility to a CSV file for a specific sector with stock and sector names.
#     """

#     id_to_ticker = {v: k for k, v in ticker_to_id.items()}

#     # Map ticker_ids to actual ticker names using ticker_mapping
#     ticker_names = [id_to_ticker[ticker_id] for ticker_id in test_tickers]

#     # Map sector_id to sector name using sector_mapping
#     sector_name = sector_mapping.get(sector_id, "Unknown Sector")

#     # Create DataFrame to store predictions
#     df = pd.DataFrame({
#         'Stock': ticker_names,
#         'Log_Returns': log_returns,
#         'Volatility': volatility,
#         'Sector': [sector_name] * len(test_tickers)  # Add the sector name instead of sector_id
#     })

#     # Check if the file already exists to determine if headers should be written
#     file_exists = os.path.exists(output_filename)

#     # Append to the CSV file (with headers only if the file doesn't exist)
#     df.to_csv(output_filename, index=False, mode='a', header=not file_exists)  # Only write header if the file doesn't exist

#     print(f"Results for sector '{sector_name}' saved to {output_filename}")

# health_indices = [i for i, sector in enumerate(train_sectors) if sector == 1]

# X_health = [X_train[i] for i in health_indices]
# y_health = [y_train[i] for i in health_indices]
# tickers_health = [train_tickers[i] for i in health_indices]

# log_returns_health = predict_logreturns(sector_models[1], X_health, tickers_health, sector_id=1, scalers=scalers)

# # Calculate volatility
# volatility_health = calculate_volatility(log_returns_health)

# # Save to CSV
# output_filename = 'healthcare_sector_predictions.csv'
# save_predictions_to_csv(sector_id=1, log_returns=log_returns_health, volatility=volatility_health, test_tickers=tickers_health, output_filename=output_filename, ticker_mapping=ticker_mapping, sector_mapping=sector_mapping)



# #-x-x-x-x-x-x-x-x-x-x-x-x- FINANCIAL SERVICES -x-x-x-x-x-x-x-x-x-x-x-x-

# # Reverse mapping properly
# id_to_ticker = {v: k for k, v in ticker_to_id.items()}  # use ticker_to_id

# # Filter tickers belonging to Financial Services sector
# finservice_sector_tickers = [ticker_id for ticker_id, sector_id in zip(train_tickers, train_sectors) if sector_id == 2]

# # Get stock names correctly
# finservice_stock_names = [id_to_ticker[int(ticker_id)] for ticker_id in set(finservice_sector_tickers)]

# print("Stocks being trained in Financial Service sector:")
# print(finservice_stock_names)

# params_financial_services = {
#     'input_dim': X_train.shape[2],  # Number of features
#     'hidden_dim': 64,
#     'output_dim': 1,
#     'ticker_dim': len(ticker_mapping),  # Total unique tickers
#     'embedding_dim': 8,
#     'sector_dim': len(sector_mapping),  # Total unique sectors
#     'learning_rate': 0.001,
#     'epochs': 10,
#     'batch_size': 32
# }

# # Financial Services (sector_id -> 2)
# sector_models = train_lstm_sectorwise(X_train, y_train, train_tickers, train_sectors, sector_id=2, params=params_financial_services, sector_models=sector_models)

# def predict_logreturns(sector_model, X_test, test_tickers, sector_id, scalers):
#     """
#     Predict log returns for all stocks in the given sector.
#     """
#     predictions = []

#     for i in range(len(X_test)):
#         x_seq = X_test[i]
#         ticker_id = test_tickers[i]

#         # Reinitialize hidden and cell states
#         h_prev = np.zeros((sector_model.hidden_dim, 1))
#         c_prev = np.zeros((sector_model.hidden_dim, 1))

#         # Get the prediction for this sequence
#         y_pred, _, _ = sector_model.forward(x_seq, ticker_id, sector_id, h_prev, c_prev)
#         predictions.append(y_pred.flatten()[0])

#     # Inverse scale the predictions (log returns)
#     min_target, max_target = scalers['target']
#     predictions = np.array(predictions) * (max_target - min_target) + min_target

#     return predictions

# def calculate_volatility(log_returns, window_size=10):
#     """
#     Calculate volatility (standard deviation) of log returns for a given window size.
#     """
#     volatility = []
#     for i in range(len(log_returns)):
#         start = max(0, i - window_size + 1)
#         window = log_returns[start:i+1]
#         volatility.append(np.std(window))  # Standard deviation as volatility
#     return np.array(volatility)

# import os
# def save_predictions_to_csv(sector_id, log_returns, volatility, test_tickers, output_filename, ticker_mapping, sector_mapping):
#     """
#     Save log returns and volatility to a CSV file for a specific sector with stock and sector names.
#     """

#     id_to_ticker = {v: k for k, v in ticker_to_id.items()}

#     # Map ticker_ids to actual ticker names using ticker_mapping
#     ticker_names = [id_to_ticker[ticker_id] for ticker_id in test_tickers]

#     # Map sector_id to sector name using sector_mapping
#     sector_name = sector_mapping.get(sector_id, "Unknown Sector")

#     # Create DataFrame to store predictions
#     df = pd.DataFrame({
#         'Stock': ticker_names,
#         'Log_Returns': log_returns,
#         'Volatility': volatility,
#         'Sector': [sector_name] * len(test_tickers)  # Add the sector name instead of sector_id
#     })

#     # Check if the file already exists to determine if headers should be written
#     file_exists = os.path.exists(output_filename)

#     # Append to the CSV file (with headers only if the file doesn't exist)
#     df.to_csv(output_filename, index=False, mode='a', header=not file_exists)  # Only write header if the file doesn't exist

#     print(f"Results for sector '{sector_name}' saved to {output_filename}")

# finservice_indices = [i for i, sector in enumerate(train_sectors) if sector == 2]

# X_finservice = [X_train[i] for i in finservice_indices]
# y_finservice = [y_train[i] for i in finservice_indices]
# tickers_finservice = [train_tickers[i] for i in finservice_indices]

# log_returns_finservice = predict_logreturns(sector_models[2], X_finservice, tickers_finservice, sector_id=2, scalers=scalers)

# # Calculate volatility
# volatility_finservice = calculate_volatility(log_returns_finservice)

# # Save to CSV
# output_filename = 'FinancialService_sector_predictions.csv'
# save_predictions_to_csv(sector_id=2, log_returns=log_returns_finservice, volatility=volatility_finservice, test_tickers=tickers_finservice, output_filename=output_filename, ticker_mapping=ticker_mapping, sector_mapping=sector_mapping)



# #-x-x-x-x-x-x-x-x-x-x-x-x- PHYSICAL ASSETS -x-x-x-x-x-x-x-x-x-x-x-x-

# # Reverse mapping properly
# id_to_ticker = {v: k for k, v in ticker_to_id.items()}  # use ticker_to_id

# # Filter tickers belonging to Physical Assets & Resources sector
# phyassets_sector_tickers = [ticker_id for ticker_id, sector_id in zip(train_tickers, train_sectors) if sector_id == 3]

# # Get stock names correctly
# phyassets_stock_names = [id_to_ticker[int(ticker_id)] for ticker_id in set(phyassets_sector_tickers)]

# print("Stocks being trained in Physical Assets sector:")
# print(phyassets_stock_names)

# params_physical_assets_resources = {
#     'input_dim': X_train.shape[2],  # Number of features
#     'hidden_dim': 64,
#     'output_dim': 1,
#     'ticker_dim': len(ticker_mapping),  # Total unique tickers
#     'embedding_dim': 8,
#     'sector_dim': len(sector_mapping),  # Total unique sectors
#     'learning_rate': 0.001,
#     'epochs': 10,
#     'batch_size': 32
# }

# # Physical Assets & Resources (sector_id -> 3)
# sector_models = train_lstm_sectorwise(X_train, y_train, train_tickers, train_sectors, sector_id=3, params=params_physical_assets_resources, sector_models=sector_models)

# def predict_logreturns(sector_model, X_test, test_tickers, sector_id, scalers):
#     """
#     Predict log returns for all stocks in the given sector.
#     """
#     predictions = []

#     for i in range(len(X_test)):
#         x_seq = X_test[i]
#         ticker_id = test_tickers[i]

#         # Reinitialize hidden and cell states
#         h_prev = np.zeros((sector_model.hidden_dim, 1))
#         c_prev = np.zeros((sector_model.hidden_dim, 1))

#         # Get the prediction for this sequence
#         y_pred, _, _ = sector_model.forward(x_seq, ticker_id, sector_id, h_prev, c_prev)
#         predictions.append(y_pred.flatten()[0])

#     # Inverse scale the predictions (log returns)
#     min_target, max_target = scalers['target']
#     predictions = np.array(predictions) * (max_target - min_target) + min_target

#     return predictions

# def calculate_volatility(log_returns, window_size=10):
#     """
#     Calculate volatility (standard deviation) of log returns for a given window size.
#     """
#     volatility = []
#     for i in range(len(log_returns)):
#         start = max(0, i - window_size + 1)
#         window = log_returns[start:i+1]
#         volatility.append(np.std(window))  # Standard deviation as volatility
#     return np.array(volatility)

# import os
# def save_predictions_to_csv(sector_id, log_returns, volatility, test_tickers, output_filename, ticker_mapping, sector_mapping):
#     """
#     Save log returns and volatility to a CSV file for a specific sector with stock and sector names.
#     """

#     id_to_ticker = {v: k for k, v in ticker_to_id.items()}

#     # Map ticker_ids to actual ticker names using ticker_mapping
#     ticker_names = [id_to_ticker[ticker_id] for ticker_id in test_tickers]

#     # Map sector_id to sector name using sector_mapping
#     sector_name = sector_mapping.get(sector_id, "Unknown Sector")

#     # Create DataFrame to store predictions
#     df = pd.DataFrame({
#         'Stock': ticker_names,
#         'Log_Returns': log_returns,
#         'Volatility': volatility,
#         'Sector': [sector_name] * len(test_tickers)  # Add the sector name instead of sector_id
#     })

#     # Check if the file already exists to determine if headers should be written
#     file_exists = os.path.exists(output_filename)

#     # Append to the CSV file (with headers only if the file doesn't exist)
#     df.to_csv(output_filename, index=False, mode='a', header=not file_exists)  # Only write header if the file doesn't exist

#     print(f"Results for sector '{sector_name}' saved to {output_filename}")

# phyassets_indices = [i for i, sector in enumerate(train_sectors) if sector == 3]

# X_phyassets = [X_train[i] for i in phyassets_indices]
# y_phyassets = [y_train[i] for i in phyassets_indices]
# tickers_phyassets = [train_tickers[i] for i in phyassets_indices]

# log_returns_phyassets = predict_logreturns(sector_models[3], X_phyassets, tickers_phyassets, sector_id=3, scalers=scalers)

# # Calculate volatility
# volatility_phyassets = calculate_volatility(log_returns_phyassets)

# # Save to CSV
# output_filename = 'PhyscicalAssets_sector_predictions.csv'
# save_predictions_to_csv(sector_id=3, log_returns=log_returns_phyassets, volatility=volatility_phyassets, test_tickers=tickers_phyassets, output_filename=output_filename, ticker_mapping=ticker_mapping, sector_mapping=sector_mapping)



# #-x-x-x-x-x-x-x-x-x-x-x-x- CONSUMER CYCLIC -x-x-x-x-x-x-x-x-x-x-x-x-

# # Reverse mapping properly
# id_to_ticker = {v: k for k, v in ticker_to_id.items()}  # use ticker_to_id

# # Filter tickers belonging to Consumer Cyclical sector
# concyclic_sector_tickers = [ticker_id for ticker_id, sector_id in zip(train_tickers, train_sectors) if sector_id == 4]

# # Get stock names correctly
# concyclic_stock_names = [id_to_ticker[int(ticker_id)] for ticker_id in set(concyclic_sector_tickers)]

# print("Stocks being trained in Consumer Cyclic sector:")
# print(concyclic_stock_names)

# params_consumer_cyclical = {
#     'input_dim': X_train.shape[2],  # Number of features
#     'hidden_dim': 64,
#     'output_dim': 1,
#     'ticker_dim': len(ticker_mapping),  # Total unique tickers
#     'embedding_dim': 8,
#     'sector_dim': len(sector_mapping),  # Total unique sectors
#     'learning_rate': 0.001,
#     'epochs': 10,
#     'batch_size': 32
# }

# # Consumer Cyclical (sector_id -> 4)
# sector_models = train_lstm_sectorwise(X_train, y_train, train_tickers, train_sectors, sector_id=4, params=params_consumer_cyclical, sector_models=sector_models)

# def predict_logreturns(sector_model, X_test, test_tickers, sector_id, scalers):
#     """
#     Predict log returns for all stocks in the given sector.
#     """
#     predictions = []

#     for i in range(len(X_test)):
#         x_seq = X_test[i]
#         ticker_id = test_tickers[i]

#         # Reinitialize hidden and cell states
#         h_prev = np.zeros((sector_model.hidden_dim, 1))
#         c_prev = np.zeros((sector_model.hidden_dim, 1))

#         # Get the prediction for this sequence
#         y_pred, _, _ = sector_model.forward(x_seq, ticker_id, sector_id, h_prev, c_prev)
#         predictions.append(y_pred.flatten()[0])

#     # Inverse scale the predictions (log returns)
#     min_target, max_target = scalers['target']
#     predictions = np.array(predictions) * (max_target - min_target) + min_target

#     return predictions

# def calculate_volatility(log_returns, window_size=10):
#     """
#     Calculate volatility (standard deviation) of log returns for a given window size.
#     """
#     volatility = []
#     for i in range(len(log_returns)):
#         start = max(0, i - window_size + 1)
#         window = log_returns[start:i+1]
#         volatility.append(np.std(window))  # Standard deviation as volatility
#     return np.array(volatility)

# import os
# def save_predictions_to_csv(sector_id, log_returns, volatility, test_tickers, output_filename, ticker_mapping, sector_mapping):
#     """
#     Save log returns and volatility to a CSV file for a specific sector with stock and sector names.
#     """

#     id_to_ticker = {v: k for k, v in ticker_to_id.items()}

#     # Map ticker_ids to actual ticker names using ticker_mapping
#     ticker_names = [id_to_ticker[ticker_id] for ticker_id in test_tickers]

#     # Map sector_id to sector name using sector_mapping
#     sector_name = sector_mapping.get(sector_id, "Unknown Sector")

#     # Create DataFrame to store predictions
#     df = pd.DataFrame({
#         'Stock': ticker_names,
#         'Log_Returns': log_returns,
#         'Volatility': volatility,
#         'Sector': [sector_name] * len(test_tickers)  # Add the sector name instead of sector_id
#     })

#     # Check if the file already exists to determine if headers should be written
#     file_exists = os.path.exists(output_filename)

#     # Append to the CSV file (with headers only if the file doesn't exist)
#     df.to_csv(output_filename, index=False, mode='a', header=not file_exists)  # Only write header if the file doesn't exist

#     print(f"Results for sector '{sector_name}' saved to {output_filename}")

# concyclic_indices = [i for i, sector in enumerate(train_sectors) if sector == 4]

# X_concyclic = [X_train[i] for i in concyclic_indices]
# y_concyclic = [y_train[i] for i in concyclic_indices]
# tickers_concyclic = [train_tickers[i] for i in concyclic_indices]

# log_returns_concyclic = predict_logreturns(sector_models[4], X_concyclic, tickers_concyclic, sector_id=4, scalers=scalers)

# # Calculate volatility
# volatility_concyclic = calculate_volatility(log_returns_concyclic)

# # Save to CSV
# output_filename = 'ConsumerCyclic_sector_predictions.csv'
# save_predictions_to_csv(sector_id=4, log_returns=log_returns_concyclic, volatility=volatility_concyclic, test_tickers=tickers_concyclic, output_filename=output_filename, ticker_mapping=ticker_mapping, sector_mapping=sector_mapping)



# #-x-x-x-x-x-x-x-x-x-x-x-x- INDUSTRIALS -x-x-x-x-x-x-x-x-x-x-x-x-

# # Reverse mapping properly
# id_to_ticker = {v: k for k, v in ticker_to_id.items()}  # use ticker_to_id

# # Filter tickers belonging to Industrial sector
# industrial_sector_tickers = [ticker_id for ticker_id, sector_id in zip(train_tickers, train_sectors) if sector_id == 5]

# # Get stock names correctly
# industrial_stock_names = [id_to_ticker[int(ticker_id)] for ticker_id in set(industrial_sector_tickers)]

# print("Stocks being trained in Industrial sector:")
# print(industrial_stock_names)

# params_industrials = {
#     'input_dim': X_train.shape[2],  # Number of features
#     'hidden_dim': 64,
#     'output_dim': 1,
#     'ticker_dim': len(ticker_mapping),  # Total unique tickers
#     'embedding_dim': 8,
#     'sector_dim': len(sector_mapping),  # Total unique sectors
#     'learning_rate': 0.001,
#     'epochs': 10,
#     'batch_size': 32
# }

# # Industrials (sector_id -> 5)
# sector_models = train_lstm_sectorwise(X_train, y_train, train_tickers, train_sectors, sector_id=5, params=params_industrials, sector_models=sector_models)

# def predict_logreturns(sector_model, X_test, test_tickers, sector_id, scalers):
#     """
#     Predict log returns for all stocks in the given sector.
#     """
#     predictions = []

#     for i in range(len(X_test)):
#         x_seq = X_test[i]
#         ticker_id = test_tickers[i]

#         # Reinitialize hidden and cell states
#         h_prev = np.zeros((sector_model.hidden_dim, 1))
#         c_prev = np.zeros((sector_model.hidden_dim, 1))

#         # Get the prediction for this sequence
#         y_pred, _, _ = sector_model.forward(x_seq, ticker_id, sector_id, h_prev, c_prev)
#         predictions.append(y_pred.flatten()[0])

#     # Inverse scale the predictions (log returns)
#     min_target, max_target = scalers['target']
#     predictions = np.array(predictions) * (max_target - min_target) + min_target

#     return predictions

# def calculate_volatility(log_returns, window_size=10):
#     """
#     Calculate volatility (standard deviation) of log returns for a given window size.
#     """
#     volatility = []
#     for i in range(len(log_returns)):
#         start = max(0, i - window_size + 1)
#         window = log_returns[start:i+1]
#         volatility.append(np.std(window))  # Standard deviation as volatility
#     return np.array(volatility)

# import os
# def save_predictions_to_csv(sector_id, log_returns, volatility, test_tickers, output_filename, ticker_mapping, sector_mapping):
#     """
#     Save log returns and volatility to a CSV file for a specific sector with stock and sector names.
#     """

#     id_to_ticker = {v: k for k, v in ticker_to_id.items()}

#     # Map ticker_ids to actual ticker names using ticker_mapping
#     ticker_names = [id_to_ticker[ticker_id] for ticker_id in test_tickers]

#     # Map sector_id to sector name using sector_mapping
#     sector_name = sector_mapping.get(sector_id, "Unknown Sector")

#     # Create DataFrame to store predictions
#     df = pd.DataFrame({
#         'Stock': ticker_names,
#         'Log_Returns': log_returns,
#         'Volatility': volatility,
#         'Sector': [sector_name] * len(test_tickers)  # Add the sector name instead of sector_id
#     })

#     # Check if the file already exists to determine if headers should be written
#     file_exists = os.path.exists(output_filename)

#     # Append to the CSV file (with headers only if the file doesn't exist)
#     df.to_csv(output_filename, index=False, mode='a', header=not file_exists)  # Only write header if the file doesn't exist

#     print(f"Results for sector '{sector_name}' saved to {output_filename}")

# industrial_indices = [i for i, sector in enumerate(train_sectors) if sector == 5]

# X_industrial = [X_train[i] for i in industrial_indices]
# y_industrial = [y_train[i] for i in industrial_indices]
# tickers_industrial = [train_tickers[i] for i in industrial_indices]

# log_returns_industrial = predict_logreturns(sector_models[5], X_industrial, tickers_industrial, sector_id=5, scalers=scalers)

# # Calculate volatility
# volatility_industrial = calculate_volatility(log_returns_industrial)

# # Save to CSV
# output_filename = 'Industrial_sector_predictions.csv'
# save_predictions_to_csv(sector_id=5, log_returns=log_returns_industrial, volatility=volatility_industrial, test_tickers=tickers_industrial, output_filename=output_filename, ticker_mapping=ticker_mapping, sector_mapping=sector_mapping)



# #-x-x-x-x-x-x-x-x-x-x-x-x- COMMUNICATION SERVICES -x-x-x-x-x-x-x-x-x-x-x-x-

# # Reverse mapping properly
# id_to_ticker = {v: k for k, v in ticker_to_id.items()}  # use ticker_to_id

# # Filter tickers belonging to Communication Services sector
# commserv_sector_tickers = [ticker_id for ticker_id, sector_id in zip(train_tickers, train_sectors) if sector_id == 6]

# # Get stock names correctly
# commserv_stock_names = [id_to_ticker[int(ticker_id)] for ticker_id in set(commserv_sector_tickers)]

# print("Stocks being trained in Communication Services sector:")
# print(commserv_stock_names)

# params_communication_servicess = {
#     'input_dim': X_train.shape[2],  # Number of features
#     'hidden_dim': 64,
#     'output_dim': 1,
#     'ticker_dim': len(ticker_mapping),  # Total unique tickers
#     'embedding_dim': 8,
#     'sector_dim': len(sector_mapping),  # Total unique sectors
#     'learning_rate': 0.001,
#     'epochs': 10,
#     'batch_size': 32
# }

# # Communication Services (sector_id -> 6)
# sector_models = train_lstm_sectorwise(X_train, y_train, train_tickers, train_sectors, sector_id=6, params=params_communication_servicess, sector_models=sector_models)

# def predict_logreturns(sector_model, X_test, test_tickers, sector_id, scalers):
#     """
#     Predict log returns for all stocks in the given sector.
#     """
#     predictions = []

#     for i in range(len(X_test)):
#         x_seq = X_test[i]
#         ticker_id = test_tickers[i]

#         # Reinitialize hidden and cell states
#         h_prev = np.zeros((sector_model.hidden_dim, 1))
#         c_prev = np.zeros((sector_model.hidden_dim, 1))

#         # Get the prediction for this sequence
#         y_pred, _, _ = sector_model.forward(x_seq, ticker_id, sector_id, h_prev, c_prev)
#         predictions.append(y_pred.flatten()[0])

#     # Inverse scale the predictions (log returns)
#     min_target, max_target = scalers['target']
#     predictions = np.array(predictions) * (max_target - min_target) + min_target

#     return predictions

# def calculate_volatility(log_returns, window_size=10):
#     """
#     Calculate volatility (standard deviation) of log returns for a given window size.
#     """
#     volatility = []
#     for i in range(len(log_returns)):
#         start = max(0, i - window_size + 1)
#         window = log_returns[start:i+1]
#         volatility.append(np.std(window))  # Standard deviation as volatility
#     return np.array(volatility)

# import os
# def save_predictions_to_csv(sector_id, log_returns, volatility, test_tickers, output_filename, ticker_mapping, sector_mapping):
#     """
#     Save log returns and volatility to a CSV file for a specific sector with stock and sector names.
#     """

#     id_to_ticker = {v: k for k, v in ticker_to_id.items()}

#     # Map ticker_ids to actual ticker names using ticker_mapping
#     ticker_names = [id_to_ticker[ticker_id] for ticker_id in test_tickers]

#     # Map sector_id to sector name using sector_mapping
#     sector_name = sector_mapping.get(sector_id, "Unknown Sector")

#     # Create DataFrame to store predictions
#     df = pd.DataFrame({
#         'Stock': ticker_names,
#         'Log_Returns': log_returns,
#         'Volatility': volatility,
#         'Sector': [sector_name] * len(test_tickers)  # Add the sector name instead of sector_id
#     })

#     # Check if the file already exists to determine if headers should be written
#     file_exists = os.path.exists(output_filename)

#     # Append to the CSV file (with headers only if the file doesn't exist)
#     df.to_csv(output_filename, index=False, mode='a', header=not file_exists)  # Only write header if the file doesn't exist

#     print(f"Results for sector '{sector_name}' saved to {output_filename}")

# commserv_indices = [i for i, sector in enumerate(train_sectors) if sector == 6]

# X_commserv = [X_train[i] for i in commserv_indices]
# y_commserv = [y_train[i] for i in commserv_indices]
# tickers_commserv = [train_tickers[i] for i in commserv_indices]

# log_returns_commserv = predict_logreturns(sector_models[6], X_commserv, tickers_commserv, sector_id=6, scalers=scalers)

# # Calculate volatility
# volatility_commserv = calculate_volatility(log_returns_commserv)

# # Save to CSV
# output_filename = 'CommunicationServices_sector_predictions.csv'
# save_predictions_to_csv(sector_id=6, log_returns=log_returns_commserv, volatility=volatility_commserv, test_tickers=tickers_commserv, output_filename=output_filename, ticker_mapping=ticker_mapping, sector_mapping=sector_mapping)



# #-x-x-x-x-x-x-x-x-x-x-x-x- CONSUMER DEFENCE -x-x-x-x-x-x-x-x-x-x-x-x-

# # Reverse mapping properly
# id_to_ticker = {v: k for k, v in ticker_to_id.items()}  # use ticker_to_id

# # Filter tickers belonging to Consumer Defence sector
# consdef_sector_tickers = [ticker_id for ticker_id, sector_id in zip(train_tickers, train_sectors) if sector_id == 7]

# # Get stock names correctly
# consdef_stock_names = [id_to_ticker[int(ticker_id)] for ticker_id in set(consdef_sector_tickers)]

# print("Stocks being trained in Consumer Defence sector:")
# print(consdef_stock_names)

# params_consumer_defensive = {
#     'input_dim': X_train.shape[2],  # Number of features
#     'hidden_dim': 64,
#     'output_dim': 1,
#     'ticker_dim': len(ticker_mapping),  # Total unique tickers
#     'embedding_dim': 8,
#     'sector_dim': len(sector_mapping),  # Total unique sectors
#     'learning_rate': 0.001,
#     'epochs': 10,
#     'batch_size': 32
# }

# # Consumer Defensive (sector_id -> 7)
# sector_models = train_lstm_sectorwise(X_train, y_train, train_tickers, train_sectors, sector_id=7, params=params_consumer_defensive, sector_models=sector_models)

# def predict_logreturns(sector_model, X_test, test_tickers, sector_id, scalers):
#     """
#     Predict log returns for all stocks in the given sector.
#     """
#     predictions = []

#     for i in range(len(X_test)):
#         x_seq = X_test[i]
#         ticker_id = test_tickers[i]

#         # Reinitialize hidden and cell states
#         h_prev = np.zeros((sector_model.hidden_dim, 1))
#         c_prev = np.zeros((sector_model.hidden_dim, 1))

#         # Get the prediction for this sequence
#         y_pred, _, _ = sector_model.forward(x_seq, ticker_id, sector_id, h_prev, c_prev)
#         predictions.append(y_pred.flatten()[0])

#     # Inverse scale the predictions (log returns)
#     min_target, max_target = scalers['target']
#     predictions = np.array(predictions) * (max_target - min_target) + min_target

#     return predictions

# def calculate_volatility(log_returns, window_size=10):
#     """
#     Calculate volatility (standard deviation) of log returns for a given window size.
#     """
#     volatility = []
#     for i in range(len(log_returns)):
#         start = max(0, i - window_size + 1)
#         window = log_returns[start:i+1]
#         volatility.append(np.std(window))  # Standard deviation as volatility
#     return np.array(volatility)

# import os
# def save_predictions_to_csv(sector_id, log_returns, volatility, test_tickers, output_filename, ticker_mapping, sector_mapping):
#     """
#     Save log returns and volatility to a CSV file for a specific sector with stock and sector names.
#     """

#     id_to_ticker = {v: k for k, v in ticker_to_id.items()}

#     # Map ticker_ids to actual ticker names using ticker_mapping
#     ticker_names = [id_to_ticker[ticker_id] for ticker_id in test_tickers]

#     # Map sector_id to sector name using sector_mapping
#     sector_name = sector_mapping.get(sector_id, "Unknown Sector")

#     # Create DataFrame to store predictions
#     df = pd.DataFrame({
#         'Stock': ticker_names,
#         'Log_Returns': log_returns,
#         'Volatility': volatility,
#         'Sector': [sector_name] * len(test_tickers)  # Add the sector name instead of sector_id
#     })

#     # Check if the file already exists to determine if headers should be written
#     file_exists = os.path.exists(output_filename)

#     # Append to the CSV file (with headers only if the file doesn't exist)
#     df.to_csv(output_filename, index=False, mode='a', header=not file_exists)  # Only write header if the file doesn't exist

#     print(f"Results for sector '{sector_name}' saved to {output_filename}")

# consdef_indices = [i for i, sector in enumerate(train_sectors) if sector == 7]

# X_consdef= [X_train[i] for i in consdef_indices]
# y_consdef = [y_train[i] for i in consdef_indices]
# tickers_consdef = [train_tickers[i] for i in consdef_indices]

# log_returns_consdef = predict_logreturns(sector_models[7], X_consdef, tickers_consdef, sector_id=7, scalers=scalers)

# # Calculate volatility
# volatility_consdef = calculate_volatility(log_returns_consdef)

# # Save to CSV
# output_filename = 'ConsumerDefence_sector_predictions.csv'
# save_predictions_to_csv(sector_id=7, log_returns=log_returns_consdef, volatility=volatility_consdef, test_tickers=tickers_consdef, output_filename=output_filename, ticker_mapping=ticker_mapping, sector_mapping=sector_mapping)

