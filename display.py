import numpy as np
from get_stats import lift
from get_stats import confidence as stat_confidence
from get_stats import support as stat_support


def display_frequent_itemsets(transactions, itemsets, by_size=False, sort_by_support=True, display_lift=False):
    t_len = len(transactions)
    if by_size:
        itemsets = {k: v for k, v in sorted(itemsets.items(), key=lambda item: len(item[0]))}
        sizes = list(map(len, list(itemsets.keys())))
        smallest_size = min(sizes)
        largest_size = max(sizes)
        if sort_by_support:
            for size in range(smallest_size, largest_size + 1):
                print(f"Itemsets of size {size}:")
                supports = []
                size_itemsets = []
                for itemset, support in itemsets.items():
                    if len(itemset) == size:
                        size_itemsets.append(itemset)
                        supports.append(support)
                        # print(f"Frequent Itemset: {itemset}, Support: {support}")
                order = np.argsort(supports)[::-1]
                for i in order:
                    if display_lift and len(size_itemsets[i]) >= 2:
                        lift_strings=[]
                        max_lift = 0.
                        for item in size_itemsets[i]:
                            item_lift = round(lift(transactions, itemsets, size_itemsets[i], item), 2)
                            if item_lift > max_lift:
                                max_lift = item_lift
                            if item_lift <= 1.00: continue
                            item_support = stat_support(transactions, itemsets, item)
                            # item_confidence = round(stat_confidence(item, size_itemsets[i], transactions, itemsets),2)
                            new_sell_chance = round(item_lift * min(item_support, 1), 2)
                            lift_strings.append(
                                f"\t\tTimes Likelier to Sell {item.capitalize()}: {item_lift}x (Sell Chance: {round(item_support, 2)}->{new_sell_chance})")
                        if max_lift > 1.00:
                            print(
                                f"\tFrequent Itemset: {size_itemsets[i]}, Support: {supports[i]*t_len}")
                            for lift_string in lift_strings:
                                print(lift_string)
                    else:
                        print(f"\tFrequent Itemset: {size_itemsets[i]}, Support: {supports[i]*t_len}")
                print()
        else:
            for size in range(smallest_size, largest_size + 1):
                print(f"Itemsets of size {size}:")
                for itemset, support in itemsets.items():
                    if len(itemset) == size:
                        print(f"Frequent Itemset: {itemset}, Support: {support}")
                print()
    else:
        itemsets = {k: v for k, v in sorted(itemsets.items(), key=lambda item: item[1], reverse=sort_by_support)}

        for itemset, support in itemsets.items():
            print(f"Frequent Itemset: {itemset}, Support: {support}")

def visualize_support_by_itemset_size(tree_frequent_patterns):
    import matplotlib.pyplot as plt
    tfp = tree_frequent_patterns
    tfp.plot(x='itemset_size',y='support_count',kind='scatter')
    plt.show()