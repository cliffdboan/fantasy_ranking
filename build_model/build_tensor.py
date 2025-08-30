import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_cleaning.create_df import all_data

# Make a copy to avoid modifying the original
data = all_data.copy()

# Convert target to numeric
data['FantPt'] = pd.to_numeric(data['FantPt'], errors='coerce')

# Convert Age to numeric and fill NaNs
data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
median_age = data['Age'].median()
data['Age'] = data['Age'].fillna(median_age)

# Convert key columns to numeric
data['FantPt'] = pd.to_numeric(data['FantPt'], errors='coerce')
data['G'] = pd.to_numeric(data['G'], errors='coerce')

# Sort by player and age to get chronological order
data_sorted = data.sort_values(['Player', 'Age'])

# Create year-to-year prediction pairs: current year stats -> next year fantasy points
next_year_stats = data_sorted.groupby('Player')[['FantPt', 'G']].shift(-1)
data_sorted['Next_Year_FantPt'] = next_year_stats['FantPt']

# Add trend features from previous year
prev_stats = data_sorted.groupby('Player')[['FantPt', 'G']].shift(1)
data_sorted['FantPt_Change'] = data_sorted['FantPt'] - prev_stats['FantPt']
data_sorted['Games_Change'] = data_sorted['G'] - prev_stats['G']

# Remove rows where we don't have next year's fantasy points (our target)
mask = data_sorted['Next_Year_FantPt'].notna()
data = data_sorted[mask].copy().reset_index(drop=True).fillna(0)

# Use next year's fantasy points as target
target = data.pop('Next_Year_FantPt')
data.pop('FantPt')  # Remove current year fantasy points from features

# Focus on raw stats only - NO fantasy scoring columns
performance_stats = [
    'Age', 'G', 'GS',  # Basic info
    'Cmp', 'Att', 'Yds', 'TD', 'Int',  # Passing stats
    'Tgt', 'Rec', 'Y/R',  # Receiving stats
    'Y/A',  # Rushing efficiency
    'Fmb', 'FL',  # Turnovers
    'FantPt_Change', 'Games_Change'  # Trend features
]

# Select only performance columns that exist
feature_cols = []
for col in performance_stats:
    if col in data.columns:
        feature_cols.append(col)

print(f"Available columns: {list(data.columns)}")
print(f"Selected features: {feature_cols}")

print(f"Using {len(feature_cols)} performance features for training")

# Create features dataframe with only performance stats
features_df = data[feature_cols].copy()

# Convert all to numeric and fill NaNs
for col in features_df.columns:
    features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
    features_df[col] = features_df[col].fillna(0)

# Convert to numpy arrays
X = features_df.values.astype(np.float32)
y = target.values.astype(np.float32)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data randomly (better than age-based splitting)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=None
)

# Further split training data for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# Batch and prefetch datasets
BATCH_SIZE = 32
train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Store important variables for model building
num_features = X.shape[1]
feature_names = features_df.columns.tolist()

print(f"Dataset prepared:")
print(f"- Features: {num_features}")
print(f"- Training samples: {len(X_train)}")
print(f"- Validation samples: {len(X_val)}")
print(f"- Test samples: {len(X_test)}")
print(f"- Target range: {y.min():.2f} to {y.max():.2f}")
