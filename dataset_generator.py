import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# Merchants by category
merchants = {
    'Travel': ['Uber', 'United Airlines', 'Marriott Hotel', 'Shell Gas Station'],
    'Meals': ['Starbucks', 'McDonald\'s', 'Chipotle', 'Pizza Hut'],
    'Software': ['AWS', 'OpenAI API', 'GitHub', 'Microsoft 365'],
    'Utilities': ['Electric Co', 'Water Department', 'Comcast Internet'],
    'Office': ['Staples', 'Office Depot', 'Amazon Business'],
    'Salary': ['Payroll - Employee'],
    'Miscellaneous': ['Walmart', 'Target', 'CVS Pharmacy']
}

payment_methods = ['Credit Card', 'Debit Card', 'Bank Transfer', 'Cash']
locations = ['New York, NY', 'San Francisco, CA', 'Chicago, IL', 'Austin, TX', 'Boston, MA']

data = []
start_date = datetime(2025, 1, 1)

# Generate 95 normal transactions
for i in range(95):
    cat = np.random.choice(list(merchants.keys()), p=[0.2, 0.25, 0.15, 0.1, 0.15, 0.05, 0.1])
    merchant = np.random.choice(merchants[cat])
    
    if cat == 'Travel':
        amount = round(np.random.uniform(25, 400), 2)
    elif cat == 'Software':
        amount = round(np.random.uniform(15, 150), 2)
    elif cat == 'Salary':
        amount = round(np.random.uniform(3000, 5000), 2)
    else:
        amount = round(np.random.uniform(10, 120), 2)
    
    date = start_date + timedelta(days=np.random.randint(0, 90))
    transaction_id = f"TXN{1000 + i}"
    
    data.append({
        'transaction_id': transaction_id,
        'date': date.strftime('%Y-%m-%d'),
        'merchant': merchant,
        'description': merchant,
        'amount': amount,
        'currency': 'USD',
        'payment_method': np.random.choice(payment_methods),
        'location': np.random.choice(locations),
        'notes': '' if np.random.rand() > 0.3 else 'Business expense'
    })

# ERROR TYPE 1: Missing required field (description is empty)
data.append({
    'transaction_id': 'TXN2001',
    'date': '2025-02-15',
    'merchant': 'Unknown Store',
    'description': '',
    'amount': 75.50,
    'currency': 'USD',
    'payment_method': 'Credit Card',
    'location': 'New York, NY',
    'notes': ''
})

# ERROR TYPE 2: Malformed date
data.append({
    'transaction_id': 'TXN2002',
    'date': '15-Feb-2025',
    'merchant': 'Best Buy',
    'description': 'Best Buy Electronics',
    'amount': 199.99,
    'currency': 'USD',
    'payment_method': 'Debit Card',
    'location': 'Chicago, IL',
    'notes': 'Office supplies'
})

# ERROR TYPE 3: Unusually high amount
data.append({
    'transaction_id': 'TXN2003',
    'date': '2025-02-20',
    'merchant': 'Suspicious Vendor LLC',
    'description': 'Large Equipment Purchase',
    'amount': 9500.00,
    'currency': 'USD',
    'payment_method': 'Bank Transfer',
    'location': 'Austin, TX',
    'notes': 'Urgent purchase'
})

# ERROR TYPE 4: Duplicate transactions
data.append({
    'transaction_id': 'TXN2004',
    'date': '2025-01-25',
    'merchant': 'Starbucks',
    'description': 'Starbucks Coffee',
    'amount': 45.67,
    'currency': 'USD',
    'payment_method': 'Credit Card',
    'location': 'San Francisco, CA',
    'notes': 'Team meeting'
})
data.append({
    'transaction_id': 'TXN2005',
    'date': '2025-01-25',
    'merchant': 'Starbucks',
    'description': 'Starbucks Coffee',
    'amount': 45.67,
    'currency': 'USD',
    'payment_method': 'Credit Card',
    'location': 'San Francisco, CA',
    'notes': 'Team meeting'
})

# ERROR TYPE 5: Negative amount
data.append({
    'transaction_id': 'TXN2006',
    'date': '2025-03-01',
    'merchant': 'Refund Processing',
    'description': 'Refund',
    'amount': -150.00,
    'currency': 'USD',
    'payment_method': 'Credit Card',
    'location': 'Boston, MA',
    'notes': 'Return'
})

df = pd.DataFrame(data)
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv('comprehensive_expenses.csv', index=False)

print(f"Generated {len(df)} transactions")
print(f"\nColumns ({len(df.columns)}): {list(df.columns)}")
print("\nError examples:")
print("✗ Missing description")
print("✗ Malformed date") 
print("✗ High outlier ($9,500)")
print("✗ Exact duplicate")
print("✗ Negative amount")
