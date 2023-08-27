from itertools import combinations
from collections import Counter

from tqdm import tqdm
from fpgrowth import get_fp_frequent_itemsets


# Helper function to update the candidate itemset and their support counts
def update_candidates(transactions, candidates, min_support_count):
    candidate_counts = Counter()

    for transaction in transactions:
        for candidate in candidates:
            if candidate.issubset(transaction):
                candidate_counts[candidate] += 1

    return {candidate: count for candidate, count in candidate_counts.items() if count >= min_support_count}


# Apriori algorithm implementation
def apriori(transactions, min_support, min_size=2):
    num_transactions = len(transactions)
    min_support_count = min_support * num_transactions

    # Generate initial (C1) candidate itemsets
    all_items = {frozenset([item]) for transaction in transactions for item in transaction}

    # Count support for C1 itemsets
    candidates = update_candidates(transactions, all_items, min_support_count)
    # print(candidates)
    # Variable to hold the final frequent itemsets
    final_frequent_itemsets = {k: v for k, v in candidates.items()}

    k = 2
    pbar = tqdm(total=k)
    #pbar.display('k:', 0)
    pbar.update(k)
    while candidates:
        # Generate new candidate itemsets (Ck) for next iteration
        new_candidates = set()
        for a, b in combinations(candidates.keys(), 2):
            candidate = a.union(b)
            if len(candidate) == k:
                new_candidates.add(candidate)

        # Count support for Ck itemsets and prune
        candidates = update_candidates(transactions, new_candidates, min_support_count)

        # Update final frequent itemsets
        final_frequent_itemsets.update(candidates)

        k += 1
        pbar.update(1)
    pbar.close()
    # Convert support counts to support ratio
    final_frequent_itemsets = {k: v / num_transactions for k, v in final_frequent_itemsets.items() if
                               len(k) >= min_size}

    return final_frequent_itemsets


get_frequent_itemsets = apriori #get_fp_frequent_itemsets

if __name__ == "__main__":
    from preprocess import bin_all_items
    import pickle as pkl

    transactions = pkl.load(open('transactions.pkl', 'rb'))

    print("Preprocessing transactions...", end='')
    transactions = bin_all_items(transactions, bins=10)
    print("Preprocessing complete.")
    print("Calculating...", end='')
    min_size = 1
    min_support = 0.01
    result = get_frequent_itemsets(transactions, min_support)#, min_size=min_size)
    print("Calculation complete.")
    # Print the final frequent itemsets
    # for itemset, support in result.items():
    #     print(f"Frequent Itemset: {itemset}, Support: {support}")

    from display import display_frequent_itemsets

    print("Displaying...")
    display_frequent_itemsets(transactions, result, by_size=True, sort_by_support=True, display_lift=True)
