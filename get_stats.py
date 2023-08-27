from operator import mul
from functools import reduce


def support(transactions, itemsets, items):
    # get the fraction of transactions that contain all items in the items list
    items = [items] if isinstance(items, str) else items
    if frozenset(items) in itemsets:
        return itemsets[frozenset(items)]
    return len(list(filter(lambda transaction: all(map(lambda item: item in transaction, items)), transactions))) / len(
        transactions)


def confidence(base_item, items, transactions, itemsets):
    # get the fraction of transactions containing all items divided by transactions at least containing base item
    return support(transactions, itemsets, items) / support(transactions, itemsets, [base_item])


def lift(transactions, itemsets, items, base_item):
    # supports = list(map(lambda item: support(transactions, itemsets, item), items))
    # expected_confidence = reduce(mul, supports)
    # true_confidence = confidence(base_item, items, transactions, itemsets)
    all_support = support(transactions, itemsets, items)
    item_support = support(transactions, itemsets, base_item)
    support_all_but_item = support(transactions, itemsets, list(set(items) - set([base_item])))
    # return true_confidence / expected_confidence
    return (all_support / item_support) / (support_all_but_item)
