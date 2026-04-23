import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Download stock data
print("Downloading TCS.NS data...")
df = yf.download('TCS.NS', start='2018-01-01', end='2024-01-01')

# Extract 'Close' price and drop missing values
data = df[['Close']].dropna()

# Plot raw time series
plt.figure(figsize=(15, 6))
plt.plot(data.index, data['Close'], label='TCS Close Price', color='blue')
plt.title('TCS.NS Daily Close Price (2018 - 2024)')
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.grid(True)
plt.savefig('raw_stock_data.png', bbox_inches='tight')
plt.close()

raw_values = data.values

# 80/20 Train-Test split sequentially
train_size = int(len(raw_values) * 0.8)
train_data_raw, test_data_raw = raw_values[0:train_size, :], raw_values[train_size:len(raw_values), :]

# Scaling using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data_raw)
test_scaled = scaler.transform(test_data_raw)

# Create windows
def create_dataset(dataset, window_size=10):
    X, Y = [], []
    for i in range(len(dataset) - window_size):
        X.append(dataset[i:(i + window_size), 0])
        Y.append(dataset[i + window_size, 0])
    return np.array(X), np.array(Y)

window_size = 10
X_train_full, y_train_full = create_dataset(train_scaled, window_size)
X_test, y_test = create_dataset(test_scaled, window_size)

# Baseline Models
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_full, y_train_full)
knn_pred = knn.predict(X_test)

svr = SVR(kernel='rbf', C=100, gamma='scale')
svr.fit(X_train_full, y_train_full)
svr_pred = svr.predict(X_test)

X_train_full_lstm = np.reshape(X_train_full, (X_train_full.shape[0], X_train_full.shape[1], 1))
X_test_lstm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

lstm = Sequential([
    LSTM(32, input_shape=(window_size, 1)),
    Dense(1)
])
lstm.compile(optimizer='adam', loss='mse')
lstm.fit(X_train_full_lstm, y_train_full, epochs=50, batch_size=32, verbose=0)
lstm_pred = lstm.predict(X_test_lstm, verbose=0).flatten()

# Experiments
train_sizes = [100, 300, 500, 1000, len(X_train_full)]
results = []

for size in train_sizes:
    X_train_sub = X_train_full[-size:]
    y_train_sub = y_train_full[-size:]
    
    knn.fit(X_train_sub, y_train_sub)
    knn_p = knn.predict(X_test)
    
    svr.fit(X_train_sub, y_train_sub)
    svr_p = svr.predict(X_test)
    
    X_train_sub_lstm = np.reshape(X_train_sub, (X_train_sub.shape[0], X_train_sub.shape[1], 1))
    
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)
    
    lstm_sub = Sequential([
        LSTM(32, input_shape=(window_size, 1)),
        Dense(1)
    ])
    lstm_sub.compile(optimizer='adam', loss='mse')
    lstm_sub.fit(X_train_sub_lstm, y_train_sub, epochs=50, batch_size=32, verbose=0)
    lstm_p = lstm_sub.predict(X_test_lstm, verbose=0).flatten()
    
    results.append({
        'Data Size': size,
        'KNN_MSE': mean_squared_error(y_test, knn_p),
        'SVR_MSE': mean_squared_error(y_test, svr_p),
        'LSTM_MSE': mean_squared_error(y_test, lstm_p),
        'KNN_MAE': mean_absolute_error(y_test, knn_p),
        'SVR_MAE': mean_absolute_error(y_test, svr_p),
        'LSTM_MAE': mean_absolute_error(y_test, lstm_p)
    })

results_df = pd.DataFrame(results)
results_df.to_csv('results_table.csv', index=False)

# Line Chart: Data Size vs MSE
plt.figure(figsize=(10, 6))
plt.plot(results_df['Data Size'], results_df['KNN_MSE'], marker='o', label='KNN')
plt.plot(results_df['Data Size'], results_df['SVR_MSE'], marker='s', label='SVR')
plt.plot(results_df['Data Size'], results_df['LSTM_MSE'], marker='^', label='LSTM')
plt.title('Learning Curves: MSE vs Training Data Size')
plt.xlabel('Training Data Size (# Samples)')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.grid(True)
plt.savefig('learning_curves.png')
plt.close()

# Actuals vs Predictions (Full Data)
plt.figure(figsize=(15, 6))
actuals_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
knn_pred_inv = scaler.inverse_transform(knn_pred.reshape(-1, 1))
svr_pred_inv = scaler.inverse_transform(svr_pred.reshape(-1, 1))
lstm_pred_inv = scaler.inverse_transform(lstm_pred.reshape(-1, 1))

plt.plot(actuals_inv, label='Actual Price', color='black', linewidth=2)
plt.plot(knn_pred_inv, label='KNN Predicted', alpha=0.7)
plt.plot(svr_pred_inv, label='SVR Predicted', alpha=0.7)
plt.plot(lstm_pred_inv, label='LSTM Predicted', alpha=0.7)
plt.title('Predictions vs Actuals on Test Set (Full Training Data)')
plt.xlabel('Days (Test Set)')
plt.ylabel('Price (INR)')
plt.legend()
plt.savefig('predictions_vs_actuals.png')
plt.close()

# Individual Plots
models_pred = [
    ('KNN', knn_pred_inv, 'blue'),
    ('SVR', svr_pred_inv, 'green'),
    ('LSTM', lstm_pred_inv, 'red')
]

for name, pred, color in models_pred:
    plt.figure(figsize=(15, 6))
    plt.plot(actuals_inv, label='Actual Price', color='black', linewidth=1.5)
    plt.plot(pred, label=f'{name} Predicted', alpha=0.8, color=color, linewidth=1.5)
    plt.title(f'{name} Predictions vs Actuals on Test Set (Full Training Data)')
    plt.xlabel('Days (Test Set)')
    plt.ylabel('Price (INR)')
    plt.legend()
    plt.savefig(f'{name.lower()}_predictions.png', bbox_inches='tight')
    plt.close()

print("Assets generated successfully: learning_curves.png, predictions_vs_actuals.png, results_table.csv, and individual prediction plots.")
