from itertools import combinations
from collections import Counter

import numpy as np
from tqdm import tqdm
from fptree import get_fptree_frequent_itemsets
from display import visualize_support_by_itemset_size


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
    # pbar.display('k:', 0)
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


# get_frequent_itemsets = lambda transactions, min_support_count: get_fptree_frequent_itemsets(transactions, min_support_count,as_pandas=True)#apriori #get_fp_frequent_itemsets

if __name__ == "__main__":
    from preprocess import bin_all_items
    import pickle as pkl
    import time

    tree = not True
    transactions = pkl.load(open('transactions.pkl', 'rb'))

    print("Preprocessing transactions...", end='')
    transactions = bin_all_items(transactions, bins=10)
    print("Preprocessing complete.")
    print("Calculating...", end='')
    if not tree:
        t1 = time.time()
        min_size = 1
        min_support = 0.01
        min_support_count = 1  # min_support * len(transactions)
        result = apriori(transactions, min_support, min_size=min_size)

        # Print the final frequent itemsets
        # for itemset, support in result.items():
        #     print(f"Frequent Itemset: {itemset}, Support: {support}")
        # print(result)
        t2 = time.time()
        print(f"Calculation complete. Time elapsed: {t2 - t1:.2f} seconds.")
        from display import display_frequent_itemsets, visualize_support_by_itemset_size

        print("Displaying...")
        display_frequent_itemsets(transactions, result, by_size=True, sort_by_support=True, display_lift=True)
    else:
        t1 = time.time()
        min_support_count_tree = 1
        min_support_count_pattern = 2

        tree, conditional_trees, frequent_patterns = get_fptree_frequent_itemsets(transactions, min_support_count_tree,
                                                                                  min_support_count_pattern,
                                                                                  k=-1)
        t2 = time.time()
        print("Calculation complete.")
        print(f"Calculation complete. Time elapsed: {t2 - t1:.2f} seconds.")
        # print(conditional_trees['fusion'])
        print(tree)
        # TODO: not getting correct support_counts (need to flow 'tree' counts to conditional_trees/frequent_patterns)
        print('Most frequent pattern order: 10^' +
              str(round(np.log10(frequent_patterns.iloc[0]['support_count'] / len(
                  transactions)), 2)) + f' ({frequent_patterns.iloc[0]["support_count"] / len(transactions)})')
        # order of 10^-3 right now (without TODO above addressed), apriori is about 0.05 (order of 10^-2)
        # print(tree)
        # visualize_support_by_itemset_size(frequent_patterns)

        # tree is about 3.75x faster than apriori (for 100k/1M transactions - (tree: 0.87s/7.28s; apriori: 5.88s/26.94s))
        # apriori: 17k-37.1k records per second
        # tree:    115k-137.4k records per second

        # best of tree vs worst of apriori = 7.1x faster
        # worst of tree vs best of apriori = 2.1x faster
