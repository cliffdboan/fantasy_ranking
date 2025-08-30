import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_cleaning.create_df import all_data

def analyze_fantasy_data():
    """Comprehensive analysis of fantasy football data"""

    # Make a copy of the data
    data = all_data.copy()

    # Convert fantasy points to numeric
    data['FantPt'] = pd.to_numeric(data['FantPt'], errors='coerce')

    print("=== FANTASY FOOTBALL DATA ANALYSIS ===\n")

    # Basic info
    print(f"Dataset shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"Fantasy points range: {data['FantPt'].min():.1f} to {data['FantPt'].max():.1f}")

    # Position analysis
    if 'FantPos' in data.columns:
        print(f"\nPosition distribution:")
        pos_counts = data['FantPos'].value_counts()
        print(pos_counts)

        # Fantasy points by position
        print(f"\nAverage fantasy points by position:")
        pos_fantasy = data.groupby('FantPos')['FantPt'].agg(['mean', 'std', 'count']).round(2)
        print(pos_fantasy)

    # Age analysis
    if 'Age' in data.columns:
        data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
        print(f"\nAge distribution:")
        print(f"Mean age: {data['Age'].mean():.1f}")
        print(f"Age range: {data['Age'].min():.0f} to {data['Age'].max():.0f}")

    # Missing data analysis
    print(f"\nMissing data analysis:")
    missing_data = data.isnull().sum()
    missing_percent = (missing_data / len(data)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_percent
    }).sort_values('Missing Percentage', ascending=False)

    print(missing_df[missing_df['Missing Count'] > 0].head(10))

    # Correlation analysis for numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        print(f"\nTop correlations with fantasy points:")
        correlations = data[numeric_cols].corr()['FantPt'].abs().sort_values(ascending=False)
        print(correlations.head(10))

    # Create visualizations
    plt.figure(figsize=(15, 10))

    # Fantasy points distribution
    plt.subplot(2, 3, 1)
    data['FantPt'].hist(bins=50, alpha=0.7)
    plt.title('Fantasy Points Distribution')
    plt.xlabel('Fantasy Points')
    plt.ylabel('Frequency')

    # Fantasy points by position
    if 'FantPos' in data.columns:
        plt.subplot(2, 3, 2)
        data.boxplot(column='FantPt', by='FantPos', ax=plt.gca())
        plt.title('Fantasy Points by Position')
        plt.suptitle('')  # Remove default title

    # Age vs Fantasy Points
    if 'Age' in data.columns:
        plt.subplot(2, 3, 3)
        plt.scatter(data['Age'], data['FantPt'], alpha=0.5)
        plt.title('Age vs Fantasy Points')
        plt.xlabel('Age')
        plt.ylabel('Fantasy Points')

    # Top correlations heatmap
    if len(numeric_cols) > 5:
        plt.subplot(2, 3, 4)
        top_corr_cols = correlations.head(6).index
        corr_matrix = data[top_corr_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Top Feature Correlations')

    # Missing data visualization
    plt.subplot(2, 3, 5)
    missing_top = missing_percent[missing_percent > 0].head(10)
    missing_top.plot(kind='bar')
    plt.title('Missing Data by Column')
    plt.ylabel('Missing Percentage')
    plt.xticks(rotation=45)

    # Fantasy points over time (if year data available)
    if any('year' in col.lower() or 'season' in col.lower() for col in data.columns):
        plt.subplot(2, 3, 6)
        # This would need to be customized based on your actual year column
        plt.title('Fantasy Points Over Time')

    plt.tight_layout()
    plt.savefig('./model_analysis/data_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    return data

if __name__ == "__main__":
    analyzed_data = analyze_fantasy_data()
    print("\nAnalysis complete! Check './model_analysis/data_analysis.png' for visualizations.")
