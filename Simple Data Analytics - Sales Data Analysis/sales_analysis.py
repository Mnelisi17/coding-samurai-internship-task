import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# If you want to run regression, install scikit-learn: pip install scikit-learn

# Load the CSV file
data = pd.read_csv('sales_data.csv')

# Show first few rows
print("Data preview:")
print(data.head())

# Basic descriptive statistics
print("\nDescriptive statistics:")
print(data.describe())

# ---------------------------
# 1) Bar chart: Total sales by product
# ---------------------------
if 'Product' in data.columns and 'Sales' in data.columns:
    sales_by_product = data.groupby('Product')['Sales'].sum()
    plt.figure()
    sales_by_product.plot(kind='bar')
    plt.title('Total Sales by Product')
    plt.ylabel('Sales')
    plt.xlabel('Product')
    plt.tight_layout()
    plt.savefig('total_sales_by_product.png', dpi=300)
    plt.show()
else:
    print("Missing 'Product' or 'Sales' columns for bar chart.")

# ---------------------------
# 2) Pie chart: Sales distribution by region
# ---------------------------
if 'Region' in data.columns and 'Sales' in data.columns:
    sales_by_region = data.groupby('Region')['Sales'].sum()
    plt.figure()
    sales_by_region.plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title('Sales Distribution by Region')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('sales_distribution_by_region.png', dpi=300)
    plt.show()
else:
    print("Missing 'Region' or 'Sales' columns for pie chart.")

# ---------------------------
# 3) Scatter plot: Advertising Spend vs Sales (and save)
# ---------------------------
if 'Advertising Spend' in data.columns and 'Sales' in data.columns:
    plt.figure()
    plt.scatter(data['Advertising Spend'], data['Sales'], color='red')
    plt.title('Advertising Spend vs Sales')
    plt.xlabel('Advertising Spend')
    plt.ylabel('Sales')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('scatter_ad_spend_vs_sales.png', dpi=300)
    plt.show()
else:
    print("Missing 'Advertising Spend' or 'Sales' columns for scatter plot.")

# ---------------------------
# 4) Correlation between Sales and Advertising Spend
# ---------------------------
if 'Advertising Spend' in data.columns and 'Sales' in data.columns:
    correlation = data['Advertising Spend'].corr(data['Sales'])
    print(f"\nCorrelation between Advertising Spend and Sales: {correlation:.2f}")
    if correlation > 0.7:
        print("=> Strong positive relationship: More advertising tends to increase sales.")
    elif correlation > 0.3:
        print("=> Moderate positive relationship: Advertising somewhat influences sales.")
    else:
        print("=> Weak relationship: Advertising spend does not strongly affect sales.")
else:
    correlation = None

# ---------------------------
# 5) Optional: Simple Linear Regression (Advertising Spend -> Sales)
# ---------------------------
try:
    from sklearn.linear_model import LinearRegression

    if 'Advertising Spend' in data.columns and 'Sales' in data.columns:
        X = data[['Advertising Spend']].values
        y = data['Sales'].values
        model = LinearRegression().fit(X, y)
        r2 = model.score(X, y)
        coef = model.coef_[0]
        intercept = model.intercept_
        print(f"\nLinear regression results:")
        print(f"  Coefficient (slope): {coef:.2f}")
        print(f"  Intercept: {intercept:.2f}")
        print(f"  R-squared: {r2:.2f}")

        # Plot regression line with scatter
        xs = np.linspace(X.min(), X.max(), 100)
        ys = model.predict(xs.reshape(-1, 1))
        plt.figure()
        plt.scatter(X, y, color='red')
        plt.plot(xs, ys, linewidth=2)
        plt.title('Advertising Spend vs Sales — with Regression Line')
        plt.xlabel('Advertising Spend')
        plt.ylabel('Sales')
        plt.tight_layout()
        plt.savefig('regression_ad_spend_vs_sales.png', dpi=300)
        plt.show()
    else:
        print("Missing 'Advertising Spend' or 'Sales' columns for regression.")
except ImportError:
    print("\nscikit-learn not installed. To run regression install it with:")
    print("pip install scikit-learn")

# ---------------------------
# 6) Quick textual summary you can copy into a report
# ---------------------------
print("\n---- Quick summary (copy into your report) ----")
if 'Product' in data.columns and 'Sales' in data.columns:
    top_product = data.groupby('Product')['Sales'].sum().idxmax()
    top_product_sales = data.groupby('Product')['Sales'].sum().max()
    print(f"- Top product by sales: {top_product} (total sales = {top_product_sales})")
if 'Region' in data.columns and 'Sales' in data.columns:
    top_region = data.groupby('Region')['Sales'].sum().idxmax()
    top_region_sales = data.groupby('Region')['Sales'].sum().max()
    print(f"- Top region by sales: {top_region} (total sales = {top_region_sales})")
if correlation is not None:
    print(f"- Correlation between Advertising Spend and Sales: {correlation:.2f}")
if 'Advertising Spend' in data.columns and 'Sales' in data.columns:
    print("- Suggestion: Consider increasing advertising spend on products/regions with proven positive relationship to boost sales.")
print("---- End summary ----")
