import numpy as np
import random
from tqdm import tqdm

def generate_random_base_probabilities(items, prob_range=(0.1, 0.9)):
    """
    Generate random base probabilities for items.

    Parameters:
        - items: List of possible items.
        - prob_range: A tuple (min_prob, max_prob) specifying the range of probabilities.

    Returns:
        - A dictionary mapping each item to its random base probability.
    """

    base_probabilities = {}

    for item in items:
        base_probabilities[item] = random.uniform(prob_range[0], prob_range[1])

    return base_probabilities


def generate_random_conditionals(items, max_conditionals_per_item, prob_range=(0.1, 0.9)):
    """
    Generate random conditional probabilities between items.

    Parameters:
        - items: List of possible items.
        - max_conditionals_per_item: The maximum number of conditional probabilities for each item.
        - prob_range: A tuple (min_prob, max_prob) specifying the range of probabilities.

    Returns:
        - A dictionary containing the random conditional probabilities.
    """
    conditionals = {}

    for item in items:
        num_conditionals = random.randint(1, max_conditionals_per_item)
        other_items = random.sample([x for x in items if x != item], num_conditionals)

        conditionals[item] = {}
        for other_item in other_items:
            conditionals[item][other_item] = random.uniform(prob_range[0], prob_range[1])

    return conditionals


def get_fake_transactions(items, min_trans=1, max_trans=5, n_transactions=1000, max_quantity=100):
    # List of items to include in fake transactions
    # items = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew", "kiwi", "lemon"]

    # max quantity for each item
    # quantities = [random.randint(max_trans) for _ in items]

    # Initialize empty list to hold all transactions
    transactions = []

    # Generate transactions
    for _ in range(n_transactions):
        # Generate a random transaction length between 1 and 5
        trans_len = random.randint(min_trans, max_trans)

        # Randomly sample 'trans_len' items from the item list without replacement
        transaction = random.sample(items, trans_len)
        transaction_quantities = random.sample(range(max_quantity), trans_len)
        transaction = dict(zip(transaction, transaction_quantities))
        # Append this transaction to the list of all transactions
        transactions.append(transaction)
    return transactions


def get_fake_transactions_conditional(items, n_transactions=1000, conditionals_per_item=4,
                                      conditionals_range=(0.1, 0.4), max_quantity=100):
    # List of items to include in fake transactions
    # items = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew", "kiwi", "lemon"]

    # max quantity for each item
    # quantities = [random.randint(max_trans) for _ in items]

    # Initialize empty list to hold all transactions
    transactions = []
    base_probabilities = generate_random_base_probabilities(items, prob_range=conditionals_range)
    conditionals = generate_random_conditionals(items, conditionals_per_item, conditionals_range)
    # Generate transactions
    for _ in tqdm(range(n_transactions)):
        # Generate a random transaction length between 1 and 5
        transaction = {}
        for item in items:
            # Check if the item appears based on its base probability
            if random.random() < base_probabilities.get(item, 0)/100:
                transaction[item] = random.randint(1, max_quantity)
        if len(transaction) == 0:
            transaction[np.random.choice(items,1)[0]] = random.randint(1, max_quantity)
            # Modify quantities based on conditional probabilities
        for item, conditional_probs in conditionals.items():
            if item in transaction:
                for other_item, prob in conditional_probs.items():
                    if random.random() < prob:
                        transaction[other_item] = random.randint(1, max_quantity)

        transactions.append(transaction)
        #print(len(transaction))
    return transactions


# Show first 5 transactions to verify
if __name__ == "__main__":
    # transactions = get_fake_transactions(['elbow', 'pipe', 'tee', 'flange', 'valve', 'fusion', 'gasket', 'pump'],
    #                                      n_transactions=100000)
    product_list = ['elbow', 'pipe', 'tee', 'flange', 'valve', 'fusion', 'gasket', 'pump', 'billy goat', 'television',
                    'computer',
                    'RC car', 'jungle gym',
                    'hasbro electronics musical radiovision display set charger port nitro gelding']
    #product_list = range(1, 1000)
    transactions = get_fake_transactions_conditional(product_list,
                                                     n_transactions=100000)
    import pickle as pkl

    pkl.dump(transactions, open('transactions.pkl', 'wb'))
