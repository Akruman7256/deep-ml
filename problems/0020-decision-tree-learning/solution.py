import math
from collections import Counter

def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    # Your code here
    if not labels:
        return 0.0

    labels_counters = Counter(labels)
    """
    Stores unique values of the labels
    directory and pairs them up with their respective key-pairs and values
    """
    total_count = len(labels)
    entropy = 0.0

    for count in labels_counters.values():            # Returns frequency of any class
        probability = count/total_count
        entropy -= probability * math.log2(probability)         #$H(S) = -\sum p_i \log_2(p_i)$
    
    return entropy

def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    """Calculate the information gain of splitting on attr."""
    # Your code here
    #$IG(S, A) = H(S) - \sum \left(\frac{\vert{}S_v\vert{}}{\vert{}S\vert{}} H(S_v)\right)$
    total_length = len(examples)
    if total_length ==0:
        return 0.0

    parent_labels = [ex[target_attr] for ex in examples]
    parent_entropy = calculate_entropy(parent_labels)

    subsets = {}
    for ex in examples:
        val = ex[attr]
        if val not in subsets:
            subsets[val] = []
        subsets[val].append(ex)

    weighted_child_entropy = 0.0
    for val, subset in subsets.items():
        subset_labels = [ex[target_attr] for ex in subset]
        weight = len(subset) / total_length
        weighted_child_entropy += weight * calculate_entropy(subset_labels)

    # 4. Information Gain = Parent Entropy - Weighted Child Entropy
    return parent_entropy - weighted_child_entropy    

def majority_class(examples: list[dict], target_attr: str) -> str:
    # 1. Pull out all labels from the list of dicts
    labels = [ex[target_attr] for ex in examples]
    # 2. Count them
    counts = Counter(labels)
    # 3. Find the highest frequency number
    max_freq = max(counts.values())
    # 4. Get all classes that hit that max frequency
    tied_classes = [cls for cls, cnt in counts.items() if cnt == max_freq]
    # 5. Sort alphabetically to break ties, pick index 0
    tied_classes.sort()
    return tied_classes[0]

def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    """Build a decision tree using the ID3 algorithm."""
    # 1. Base Case: Pure leaf
    labels = [ex[target_attr] for ex in examples]
    if len(set(labels)) == 1:
        return labels[0]

    # 2. Base Case: Out of features to split on
    if not attributes:
        return majority_class(examples, target_attr)

    # 3. Find attribute with max Information Gain
    # max() preserves the first attribute in case of equal gain
    best_attr = max(attributes, key=lambda a: calculate_information_gain(examples, a, target_attr))

    tree = {best_attr: {}}
    remaining_attrs = [a for a in attributes if a != best_attr]

    # 4. Get sorted unique values of best_attr
    attr_values = sorted(list(set(ex[best_attr] for ex in examples)))

    # 5. Build branches recursively
    for val in attr_values:
        sub_examples = [ex for ex in examples if ex[best_attr] == val]
        if not sub_examples:
            tree[best_attr][val] = majority_class(examples, target_attr)
        else:
            tree[best_attr][val] = learn_decision_tree(sub_examples, remaining_attrs, target_attr)

    return tree