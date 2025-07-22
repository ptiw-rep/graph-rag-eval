def calculate_recall_score(matched_relations, partial_matches, total_source_facts, weights):
    """
    Calculate recall score.
    
    Args:
        matched_relations: List of facts with matching relations
        partial_matches: List of facts where only subject-object exists
        total_source_facts: Total number of facts in source
        weights: dict with 'K' and 'c'
    
    Returns:
        recall_score, max_possible_score
    """
    K = weights['K']
    c = weights['c']

    score = len(matched_relations) * (c * K) + len(partial_matches) * K
    max_score = total_source_facts * (c * K)

    return score, max_score


def calculate_precision_score(correct_relations, partial_correct, hallucinations, weights):
    """
    Calculate precision score.
    
    Args:
        correct_relations: Facts with correct relation
        partial_correct: Facts with correct nodes but incorrect relation
        hallucinations: Facts not present at all in source
        weights: dict with 'L', 'm', 'N'
    
    Returns:
        precision_score, max_possible_score
    """
    L = weights['L']
    m = weights['m']
    N = weights['N']

    score = (
        len(correct_relations) * (m * L) +
        len(partial_correct) * L -
        len(hallucinations) * N
    )
    max_score = (len(correct_relations) + len(partial_correct) + len(hallucinations)) * (m * L)

    return score, max_score


def normalize_score(score, max_score):
    """Normalize score between 0 and 1"""
    if max_score == 0:
        return 0.0
    return max(score, 0) / max_score if max_score > 0 else 0.0