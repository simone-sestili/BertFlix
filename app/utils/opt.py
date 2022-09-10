def fair_sequence_division_order(seq: list, target: float, alpha: float = 0.7) -> list:
    """
    This function performs an optimization for the following scenario:
    It is given an ordered sequence of numbers and a target value, the optimization
    problem is to find the grouping of the elements of the list that minimizes
    the overall distance (squared for sign invariance) between the sum of values 
    in a group and the target. Moreover, the groups can be made by adjacent values only.
    
    This algorithm creates groups by placing divisors in the sequence, in particular
    a divisor between seq[i] and seq[i+1] has value i. The divisors are initialized
    through a greedy procedure and then moved one-by-one to improve the result
    following a least discrepancy principle.
    """
    divs = [-1]
    # greedy initialization of divisors positions
    cnt = 0
    for i in range(len(seq)-1):
        cnt += seq[i]
        if cnt >= target * alpha:
            divs.append(i)
            cnt = 0
    if seq[-1] >= target * alpha and len(seq)-1 not in divs:
        divs.append(len(seq)-1)
    divs.append(len(seq))  # there needs to be a dummy div before the sequence and a dummy div after the sequence
    
    def distance(a: int, b: int) -> float:
        # distance between target and sum of elements between index a and index b
        subseq = [seq[i] for i in range(len(seq)) if a < i <= b]
        if len(subseq) == 0:
            return 0  # empty subset
        else:
            return (target - sum(subseq))**2
    
    # least discrepancy
    improved = True  # perform first cycle
    while improved:
        improved = False
        for k in range(1, len(divs)-1):
            # check if moving div by 1 place can improve the couple
            current_score = distance(divs[k-1], divs[k]) + distance(divs[k], divs[k+1])
            # move backwards
            score = distance(divs[k-1], divs[k]-1) + distance(divs[k]-1, divs[k+1])
            if score < current_score:
                divs[k] -= 1
                improved = True
                continue
            # move forwards
            score = distance(divs[k-1], divs[k]+1) + distance(divs[k]+1, divs[k+1])
            if score < current_score:
                divs[k] += 1
                improved = True
                continue
    
    # create groups with optimized divisors
    out = []
    for k in range(1, len(divs)):
        out.append([seq[i] for i in range(len(seq)) if divs[k-1] < i <= divs[k]])
    
    return [ls for ls in out if len(ls) > 0]
