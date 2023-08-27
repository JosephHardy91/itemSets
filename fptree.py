"""
The frequent pattern growth method lets us find the frequent pattern without candidate generation.

Let us see the steps followed to mine the frequent pattern using frequent pattern growth algorithm:

#1) The first step is to scan the database to find the occurrences of the itemsets in the database. This step is the same as the first step of Apriori. The count of 1-itemsets in the database is called support count or frequency of 1-itemset.

#2) The second step is to construct the FP tree. For this, create the root of the tree. The root is represented by null.

#3) The next step is to scan the database again and examine the transactions. Examine the first transaction and find out the itemset in it. The itemset with the max count is taken at the top, the next itemset with lower count and so on. It means that the branch of the tree is constructed with transaction itemsets in descending order of count.

#4) The next transaction in the database is examined. The itemsets are ordered in descending order of count. If any itemset of this transaction is already present in another branch (for example in the 1st transaction), then this transaction branch would share a common prefix to the root.

This means that the common itemset is linked to the new node of another itemset in this transaction.

#5) Also, the count of the itemset is incremented as it occurs in the transactions. Both the common node and new node count is increased by 1 as they are created and linked according to transactions.

#6) The next step is to mine the created FP Tree. For this, the lowest node is examined first along with the links of the lowest nodes. The lowest node represents the frequency pattern length 1. From this, traverse the path in the FP Tree. This path or paths are called a conditional pattern base.

Conditional pattern base is a sub-database consisting of prefix paths in the FP tree occurring with the lowest node (suffix).

#7) Construct a Conditional FP Tree, which is formed by a count of itemsets in the path. The itemsets meeting the threshold support are considered in the Conditional FP Tree.

#8) Frequent Patterns are generated from the Conditional FP Tree.
"""
from collections import defaultdict
import numpy as np


class Tree():
    def __init__(self):
        self.root = Node(None, 0)

    def add_transaction(self, sorted_items):
        self.root.add_children(sorted_items)

    # def __str__(self):
    #     return self.root.__str__(depth=0)

    def __str__(self):
        return self.root.__str__(depth=0, prefix='', last=True)
class Node():
    def __init__(self, item, count):
        self.nodes = {}
        self.item = item
        self.count = count

    # def __str__(self, depth):
    #     #tabs = '\t' * depth
    #     #stars = '*' * depth
    #     lines = '_' * depth
    #     if len(self.nodes) == 0:
    #         return f"|{lines} {self.item}:{self.count}"
    #     else:
    #         children_str = "\n".join(
    #             node.__str__(depth=depth + 1) for node in self.nodes.values())
    #         return f"|{lines} {self.item}:{self.count}\n{children_str}"
    def __str__(self, depth=0, prefix='', last=True):
        lines = []
        connector = '└── ' if last else '├── '
        new_prefix = prefix + ('    ' if last else '│   ')

        lines.append(f"{prefix}{connector}{self.item}:{self.count}")

        child_count = len(self.nodes)
        for i, (item, node) in enumerate(self.nodes.items()):
            child_last = i == child_count - 1
            lines.append(node.__str__(depth=depth + 1, prefix=new_prefix, last=child_last))

        return '\n'.join(lines)


    def add_node(self, item, count):
        node = Node(item, count)
        self.nodes[node.item] = node
        return node

    def add_children(self, items):
        if len(items) == 0: return
        first_item = items[0]
        if first_item in self.nodes:
            self.nodes[first_item].count += 1
        else:
            self.nodes[first_item] = Node(first_item, 1)

        self.nodes[first_item].add_children(items[1:])


def get_itemcounts(transactions, min_support_count=2):
    itemcounts = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            itemcounts[item] += 1
    # print(itemcounts)
    return {k: v for k, v in itemcounts.items() if v >= min_support_count}


def get_sorted_itemcounts(itemcounts):
    sorted_itemcounts = sorted(itemcounts.items(), key=lambda x: x[1], reverse=True)
    item_indices = {item[0]: i for i, item in enumerate(sorted_itemcounts)}
    return sorted_itemcounts, item_indices


def build_tree(transactions, min_support_count=2):
    sorted_itemcounts, item_indices = get_sorted_itemcounts(
        get_itemcounts(transactions, min_support_count=min_support_count))
    # print(sorted_itemcounts)
    tree = Tree()
    for transaction in transactions:
        transaction_list = np.array(list(transaction))
        transaction_item_priorities = [item_indices[item] for item in transaction_list]
        # priority_to_index_mapping = {priority: i for i, priority in enumerate(transaction_item_priorities)}
        # sorted_priorities = sorted(transaction_item_priorities,reverse=True)
        # print(transaction_list, transaction_item_priorities, np.argsort(transaction_item_priorities)[::-1])
        sorted_items = transaction_list[np.argsort(transaction_item_priorities)[::-1]]
        tree.add_transaction(sorted_items)

    return tree


if __name__ == "__main__":
    # load transactions.pkl and build tree
    transactions = np.load("transactions.pkl", allow_pickle=True)
    tree = build_tree(transactions)
    print(tree)
