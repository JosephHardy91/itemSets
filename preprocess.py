# bin item quantity in each transaction according to the distribution for item quantities over all transactions
import numpy as np
from tqdm import tqdm

def get_item_bins(transactions, item, bins=10):
    item_quantities = []
    for transaction in transactions:
        if item in transaction:
            item_quantities.append(transaction[item])
    # bin item quantities
    bins = np.histogram(item_quantities, bins=bins)
    return bins[1]


def bin_all_items(transactions, bins=10, return_transactions=True):
    item_quantities = {}
    item_bins = {}

    new_transactions = []
    for transaction in tqdm(transactions):
        new_transaction = {}
        for item in transaction:
            if item not in item_quantities:
                item_quantities[item] = []
                item_bins[item] = get_item_bins(transactions, item, bins=bins)
            # print(item_bins[item])
            if item not in new_transaction:
                new_transaction[item] = None
            new_transaction[item] = np.searchsorted(item_bins[item], transaction[item], side='left')
        new_transactions.append(new_transaction)
    return new_transactions


if __name__ == "__main__":
    import pickle as pkl

    transactions = pkl.load(open('transactions.pkl', 'rb'))
    transactions = bin_all_items(transactions)
    print(transactions)
