#Write a set of functions and classes to use the fp growth algorithm to get the frequent itemsets.
from collections import defaultdict


# The function get_fp_frequent_itemsets() should take a list of transactions and a minimum support threshold as input and return a dictionary of frequent itemsets and their support ratios. The function should use the FP Growth algorithm to find the frequent itemsets.
def get_fp_frequent_itemsets(transactions, min_support):
    #fill in code
    min_support_count = len(transactions) * min_support
    root, header = build_tree(transactions, min_support_count)
    frequent_itemsets = {}
    find_frequent_itemsets(root, [], header, min_support_count, frequent_itemsets, len(transactions))

    # print("Frequent Itemsets:")
    # for itemset, count in frequent_itemsets.items():
    #     print(f"{set(itemset)}: {count}")

    return frequent_itemsets

def build_tree(transactions, min_support_count):
    root = Node()
    header = {}

    for transaction in transactions:
        node = root
        for item in transaction:
            if item not in node.children:
                child = Node(item)
                child.parent = node
                node.children[item] = child

                if item not in header:
                    header[item] = [child]
                else:
                    header[item].append(child)

            node = node.children[item]
        node.count += 1

    return root, header

class Node:
    def __init__(self, item=None, count=0):
        self.item = item
        self.count = count
        self.parent = None
        self.children = {}

def find_frequent_itemsets(node, path, header, min_support, frequent_itemsets, total_transactions):
    if node.count >= min_support:
        support = node.count / total_transactions
        frequent_itemset = frozenset(path)
        frequent_itemsets[frequent_itemset] = support  # node.count

        for item, nodes in header.items():
            conditional_tree = defaultdict(int)

            for leaf in nodes:
                count = leaf.count
                conditional_path = []

                parent = leaf.parent
                while parent.item is not None:
                    conditional_path.append(parent.item)
                    parent = parent.parent

                for i in range(1, len(conditional_path) + 1):
                    conditional_tree[frozenset(conditional_path[:i])] += count

            conditional_tree = {itemset: count for itemset, count in conditional_tree.items() if count >= min_support}

            if conditional_tree:
                new_header = {}
                for itemset in conditional_tree.keys():
                    for item in itemset:
                        if item not in new_header:
                            new_header[item] = []

                for itemset, count in conditional_tree.items():
                    new_node = Node(itemset, count)
                    new_node.parent = node
                    node.children[itemset] = new_node

                    for item in itemset:
                        new_header[item].append(new_node)

                find_frequent_itemsets(node.children[itemset], path + [item], new_header, min_support,
                                       frequent_itemsets, total_transactions)
