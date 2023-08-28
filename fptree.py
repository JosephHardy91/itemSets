"""
https://www.softwaretestinghelp.com/fp-growth-algorithm-data-mining/

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
from itertools import combinations

import numpy as np
import pandas as pd


class Tree:
    def __init__(self, n_transactions=None):
        self.root = Node('Top of Tree', n_transactions)

    def add_transaction(self, sorted_items, with_counts=False):
        if not with_counts:
            self.root.add_children(sorted_items)
        else:
            self.root.add_children(sorted_items, with_counts=True)

    def __str__(self):
        return self.root.__str__(depth=0, prefix='', last=True)

    def get_branches_reversed(self, with_counts=False):
        return self.root.get_branches_from_leaf(with_counts=with_counts)

    def prune(self, min_support_count=2):
        self.root.prune(min_support_count)

    def get_branches_from_leaf(self):
        return self.root.get_branches_from_leaf()


class Node:
    def __init__(self, item, count, parent=None):
        self.nodes = {}
        self.item = item
        self.count = count
        self.parent = parent

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
        node = Node(item, count, parent=self)
        self.nodes[node.item] = node
        return node

    def add_children(self, items, with_counts=False):
        if len(items) == 0: return
        if not with_counts:
            first_item = items[0]
            if first_item in self.nodes:
                self.nodes[first_item].count += 1
            else:
                self.nodes[first_item] = Node(first_item, 1, parent=self)

            self.nodes[first_item].add_children(items[1:])
        else:
            first_item, first_count = items[0]
            if first_item in self.nodes:
                self.nodes[first_item].count = min(first_count,self.nodes[first_item].count)
            else:
                self.nodes[first_item] = Node(first_item, first_count, parent=self)

            self.nodes[first_item].add_children(items[1:], with_counts=True)

    def get_branches_from_leaf(self, with_counts=False):
        # starting at each leaf, get the branch of nodes leading to the root

        branches = []
        for node in self.nodes.values():
            if len(node.nodes) == 0:
                branches.append(node.get_branch(with_counts=with_counts))
            else:
                branches.extend(node.get_branches_from_leaf(with_counts=with_counts))
        return branches

    def get_branch(self, with_counts=False):
        branch = []
        node = self
        while node is not None and node.parent is not None:
            if not with_counts:
                branch.append(node.item)
            else:
                branch.append((node.item, node.count))
            node = node.parent
        return branch

    def prune(self, min_support_count):
        # prune branches from tree
        kept_nodes = []
        for node in self.nodes.values():
            if node.count >= min_support_count:
                node.prune(min_support_count)
                kept_nodes.append(node)
        self.nodes = {node.item: node for node in kept_nodes}


def get_itemcounts(transactions, min_support_count=2):
    itemcounts = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            itemcounts[item] += 1
    return {k: v for k, v in itemcounts.items() if v >= min_support_count}


def get_sorted_itemcounts(itemcounts):
    sorted_itemcounts = sorted(itemcounts.items(), key=lambda x: x[1], reverse=True)
    item_indices = {item[0]: i for i, item in enumerate(sorted_itemcounts)}
    return sorted_itemcounts, item_indices


def build_tree(transactions, min_support_count=2):
    _, item_indices = get_sorted_itemcounts(
        get_itemcounts(transactions, min_support_count=min_support_count))
    tree = Tree(len(transactions))
    for transaction in transactions:
        transaction_list = np.array(list(transaction))
        transaction_item_priorities = [item_indices[item] for item in transaction_list]
        sorted_items = transaction_list[np.argsort(transaction_item_priorities)[::-1]]
        tree.add_transaction(sorted_items)

    return tree


def build_conditionals(tree, min_support_count=2):
    branches = tree.get_branches_reversed(with_counts=True)
    conditional_trees = {}
    i = 0
    for branch in branches:
        #print(branch)
        # if i==5:
        #     print(conditional_tree)
        #     break
        conditional_tree = conditional_trees.get(branch[0][0], Tree(None))
        conditional_tree.add_transaction(branch[1:], with_counts=True)

        conditional_trees[branch[0][0]] = conditional_tree
        i += 1

    for item in conditional_trees:
        conditional_trees[item].prune(min_support_count=min_support_count)
    return conditional_trees


def get_itemcounts_tree(tree):
    itemcounts = defaultdict(int)
    # print(tree)
    for branch in tree.get_branches_reversed():
        for item in branch:
            itemcounts[item] += 1
    return itemcounts


# each pattern is a tuple of (itemset, support_count). tree.root.count is arbitrary and unreliable for support count. ensure recursion is stopped
def get_frequent_patterns(conditional_trees, k=3, min_support_count=2):
    """
    4. Mining of FP-tree is summarized below:

The lowest node item I5 is not considered as it does not have a min support count, hence it is deleted.
The next lower node is I4. I4 occurs in 2 branches , {I2,I1,I3:,I41},{I2,I3,I4:1}. Therefore considering I4 as suffix the prefix paths will be {I2, I1, I3:1}, {I2, I3: 1}. This forms the conditional pattern base.
The conditional pattern base is considered a transaction database, an FP-tree is constructed. This will contain {I2:2, I3:2}, I1 is not considered as it does not meet the min support count.
This path will generate all combinations of frequent patterns : {I2,I4:2},{I3,I4:2},{I2,I3,I4:2}
For I3, the prefix path would be: {I2,I1:3},{I2:1}, this will generate a 2 node FP-tree : {I2:4, I1:3} and frequent patterns are generated: {I2,I3:4}, {I1:I3:3}, {I2,I1,I3:3}.
For I1, the prefix path would be: {I2:4} this will generate a single node FP-tree: {I2:4} and frequent patterns are generated: {I2, I1:4}.

Item	Conditional Pattern Base	Conditional FP-tree	Frequent Patterns Generated
I4	{I2,I1,I3:1},{I2,I3:1}	{I2:2, I3:2}	{I2,I4:2},{I3,I4:2},{I2,I3,I4:2}
I3	{I2,I1:3},{I2:1}	{I2:4, I1:3}	{I2,I3:4}, {I1:I3:3}, {I2,I1,I3:3}
I1	{I2:4}	{I2:4}	{I2,I1:4}
"""
    frequent_patterns = []
    for base_item in conditional_trees:
        if len(conditional_trees[base_item].root.nodes) == 0: continue
        conditional_tree = conditional_trees[base_item]
        conditional_itemcounts = get_itemcounts_tree(conditional_tree)
        print(base_item,conditional_itemcounts)
        # generate all combinations of itemsets from conditional_itemcounts and item, with the lowest itemcount as the support for that itemset combination

        # get itemsets
        itemsets = []
        if k == -1:
            k = len(conditional_itemcounts)
        other_items = list(conditional_itemcounts.keys())
        for i in range(2, k + 1):
            itemsets += list(combinations([base_item]+other_items, i))
        # print(itemsets, conditional_itemcounts)
        # get itemsets with support count
        itemsets = [(itemset, max([conditional_itemcounts[item] for item in other_items])) for itemset in itemsets]
        itemsets = [(itemset, count) for itemset, count in itemsets if count > min_support_count]
        # add to frequent patterns
        frequent_patterns += itemsets
    return frequent_patterns


def get_fptree_frequent_itemsets(transactions, min_support_count_tree=2, min_support_count_pattern=2, k=3):
    tree = build_tree(transactions, min_support_count=min_support_count_tree)
    conditional_trees = build_conditionals(tree, min_support_count=min_support_count_tree)
    frequent_patterns = get_frequent_patterns(conditional_trees, k=k, min_support_count=min_support_count_pattern)
    frequent_patterns = pd.DataFrame(frequent_patterns, columns=['itemset', 'support_count'])
    frequent_patterns = frequent_patterns.groupby('itemset').agg({'support_count': 'sum'}).sort_values('support_count',
                                                                                             ascending=False).reset_index()
    frequent_patterns['itemset_size'] = frequent_patterns['itemset'].apply(lambda x: len(x))
    return tree, conditional_trees, frequent_patterns


if __name__ == "__main__":
    # load transactions.pkl and build tree
    transactions = np.load("transactions.pkl", allow_pickle=True)

    min_support_count_tree = 2
    min_support_count_pattern = 2

    tree, conditional_trees, frequent_patterns = get_fptree_frequent_itemsets(transactions, min_support_count_tree,
                                                                              min_support_count_pattern,
                                                                              k=-1)

    print(
        frequent_patterns)  # TODO: not getting correct support_counts (need to flow 'tree' counts to conditional_trees/frequent_patterns)
