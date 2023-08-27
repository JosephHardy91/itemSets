from collections import defaultdict, Counter


class Node:
    def __init__(self, item, count=1):
        self.item = item
        self.count = count
        self.children = {}
        self.parent = None


def build_tree(transactions, min_support):
    header = {}
    item_counts = Counter(item for transaction in transactions for item in transaction)
    frequent_items = {item for item, count in item_counts.items() if count >= min_support}

    for item in frequent_items:
        header[item] = None

    root = Node(None)

    for transaction in transactions:
        transaction = [item for item in transaction if item in frequent_items]
        transaction.sort(key=lambda item: item_counts[item], reverse=True)

        node = root
        for item in transaction:
            if item in node.children:
                child = node.children[item]
                child.count += 1
            else:
                child = Node(item)
                child.parent = node
                node.children[item] = child

                if header[item] is None:
                    header[item] = [child]
                else:
                    header[item].append(child)

            node = child

    return root, header


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


def get_fp_frequent_itemsets(transactions, min_support):
    min_support_count = len(transactions) * min_support
    root, header = build_tree(transactions, min_support_count)
    frequent_itemsets = {}
    find_frequent_itemsets(root, [], header, min_support_count, frequent_itemsets, len(transactions))

    # print("Frequent Itemsets:")
    # for itemset, count in frequent_itemsets.items():
    #     print(f"{set(itemset)}: {count}")
    return frequent_itemsets

# TO-DO: Figure out why apriori itemsets aren't matching fp itemsets from a code review
# FP itemsets are not correct, apriori are.
# This is because the FP algorithm is not correct. To correct the FP algorithm, we need to:
# 1. Sort the transactions by frequency of items - is this true?
# 2. Sort the items in each transaction by frequency
# 3. Build the tree from the sorted transactions
# 4. Build the conditional tree from the sorted transactions
# 5. Build the frequent itemsets from the conditional tree
# 6. Sort the frequent itemsets by frequency
# 7. Return the frequent itemsets


# The reason why the FP algorithm is not correct is because it is not sorting the transactions by frequency of items.
# This is causing the tree to be built incorrectly, which is causing the conditional tree to be built incorrectly,
# which is causing the frequent itemsets to be built incorrectly.
# The reason why the apriori algorithm is correct is because it is sorting the transactions by frequency of items.

#Why does the FP algorithm need sorting?
# The FP algorithm needs sorting because it is building the tree from the transactions. If the transactions are not
# sorted, then the tree will not be built correctly. The tree needs to be built from the most frequent items to the
# least frequent items. This is because the tree is built from the transactions, and the transactions need to be
# sorted by frequency of items. If the transactions are not sorted by frequency of items, then the tree will not be
# built correctly.