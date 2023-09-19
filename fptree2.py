"""
https://medium.com/@anilcogalan/fp-growth-algorithm-how-to-analyze-user-behavior-and-outrank-your-competitors-c39af08879db
"""
import numpy as np
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import pickle as pkl
import random

transactions = pkl.load(open('transactions.pkl', 'rb'))
transactions = pd.DataFrame(transactions)
transactions[~pd.isnull(transactions)] = 1
transactions[pd.isnull(transactions)] = 0

frequent_itemsets = fpgrowth(transactions, min_support=0.01, use_colnames=True)
print(frequent_itemsets)  # "it just works"

product_list = ['elbow', 'pipe', 'tee', 'flange', 'valve', 'fusion', 'gasket', 'pump', 'billy goat', 'television',
                'computer',
                'RC car', 'jungle gym',
                'hemrdscpng'] #hasbro electronics musical radiovision display set charger port nitro gelding
n = len(product_list)
dummy_transaction_dataset = [[product_list[random.randint(0, n-1)] for _ in range(1, random.randint(2, n))] for _ in
                             range(10000)]
te = TransactionEncoder()
te_array = te.fit(dummy_transaction_dataset).transform(dummy_transaction_dataset)
dummy_transaction_df = pd.DataFrame(te_array, columns=te.columns_)

frequent_itemsets = fpgrowth(dummy_transaction_df, min_support=0.1, use_colnames=True)

frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)

print(frequent_itemsets.head(50))

import matplotlib.pyplot as plt
df = frequent_itemsets
df['str_itemsets'] = df['itemsets'].apply(lambda x: ', '.join(list(x)))
# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.bar(range(len(df['str_itemsets'])), df['support'], color='blue')
plt.xlabel('Support')
plt.ylabel('Itemsets')
plt.title('Histogram of Support vs Itemsets')
plt.grid(axis='x')

# Annotate the bars with the support values
# for i, v in enumerate(df['support']):
#     plt.text(v, i, f" {v:.4f}", color='black', va='center', fontweight='bold')

plt.show()