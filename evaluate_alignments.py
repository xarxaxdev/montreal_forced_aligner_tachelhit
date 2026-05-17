def interval_overlap(interval1, interval2):
    """Check if two intervals overlap and return overlap ratio."""
    start1, end1 = interval1
    start2, end2 = interval2
    
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    
    if overlap_start >= overlap_end:
        return 0.0
    
    overlap = overlap_end - overlap_start
    len1 = end1 - start1
    len2 = end2 - start2
    avg_len = (len1 + len2) / 2
    
    return overlap / avg_len if avg_len > 0 else 0.0


def labeled_interval_levenshtein(seq1, seq2, overlap_threshold=0.5, label_weight=0.5):
    """
    Levenshtein distance for sequences of labeled intervals with float support.
    
    Args:
        seq1, seq2: Lists of tuples (start, end, label) representing labeled intervals
                    start and end can be floats
        overlap_threshold: Minimum overlap ratio to consider intervals matching (0-1)
        label_weight: Weight for label mismatch (0=ignore labels, 1=labels must match)
    
    Returns:
        Float edit distance between labeled interval sequences
    """
    m, n = len(seq1), len(seq2)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases (float costs)
    for i in range(m + 1):
        dp[i][0] = float(i)
    for j in range(n + 1):
        dp[0][j] = float(j)
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            start1, end1, label1 = seq1[i-1]
            start2, end2, label2 = seq2[j-1]
            
            overlap = interval_overlap((start1, end1), (start2, end2))
            label_match = 1.0 if label1 == label2 else 0.0
            
            # Combined cost: overlap penalty + label penalty
            if overlap >= overlap_threshold and label1 == label2:
                # Both intervals overlap and labels match
                substitution_cost = (1.0 - overlap) * (1.0 - label_weight) + (1.0 - label_match) * label_weight
            elif overlap >= overlap_threshold:
                # Intervals overlap but labels differ
                substitution_cost = (1.0 - overlap) * (1.0 - label_weight) + label_weight
            else:
                # No sufficient overlap - full substitution cost
                substitution_cost = 1.0
            
            dp[i][j] = min(
                dp[i-1][j] + 1.0,              # Deletion
                dp[i][j-1] + 1.0,              # Insertion
                dp[i-1][j-1] + substitution_cost  # Substitution/match
            )
    
    return dp[m][n]


# Example usage
if __name__ == "__main__":
    # Example: comparing labeled intervals (e.g., annotated events)
    seq1 = [(0, 5, 'A'), (6, 10, 'B'), (12, 15, 'A')]
    seq2 = [(0, 4, 'A'), (5, 11, 'B'), (13, 16, 'C')]
    
    distance = labeled_interval_levenshtein(seq1, seq2)
    print(f"Labeled interval distance: {distance}")
    
    # Example with exact matches
    seq3 = [(0, 5, 'X'), (10, 15, 'Y')]
    seq4 = [(0, 5, 'X'), (10, 15, 'Y')]
    print(f"Exact match distance: {labeled_interval_levenshtein(seq3, seq4)}")
    
    # Example: same intervals, different labels
    seq5 = [(0, 5, 'A'), (10, 15, 'B')]
    seq6 = [(0, 5, 'X'), (10, 15, 'Y')]
    print(f"Same intervals, different labels: {labeled_interval_levenshtein(seq5, seq6)}")
    
    # Example: overlapping intervals with matching vs mismatching labels
    seq7 = [(0, 10, 'A')]
    seq8 = [(2, 8, 'A')]
    seq9 = [(2, 8, 'B')]
    print(f"Overlap + same label: {labeled_interval_levenshtein(seq7, seq8)}")
    print(f"Overlap + diff label: {labeled_interval_levenshtein(seq7, seq9)}")
    
    # Example with float intervals (e.g., continuous time ranges)
    seq10 = [(0.0, 2.5, 'A'), (3.7, 8.2, 'B'), (9.1, 12.3, 'C')]
    seq11 = [(0.1, 2.4, 'A'), (3.8, 8.0, 'B'), (10.0, 12.5, 'C')]
    print(f"\nFloat intervals distance: {labeled_interval_levenshtein(seq10, seq11)}")
    
    # Example with fractional overlaps
    seq12 = [(0.0, 1.0, 'X'), (2.5, 3.5, 'Y')]
    seq13 = [(0.3, 0.9, 'X'), (2.7, 3.3, 'Y')]
    print(f"Fractional overlaps: {labeled_interval_levenshtein(seq12, seq13)}")
