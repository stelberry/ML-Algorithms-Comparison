import numpy as np

def conformity_score(X, y, X_train, y_train):
    
    """
    Conformity measure: alpha = d_diff / d_same
    where d_diff = distance to nearest different-class neighbor
    and d_same = distance to nearest same-class neighbor
    """
    
    # create empty lists to store distances and class info    
    distances = []
    same_class = []
    
    # loop through every point in the training data
    for i in range(len(X_train)):
        # calculate the distance from our point X to this training point
        dist = np.linalg.norm(X - X_train[i]) 
        
        if dist > 0:  # ignore the point itself if it's in training set
            # store the distance 
            distances.append(dist)
            # store whether this training point has the same label as our point X
            same_class.append(y_train[i] == y)
    
    # initialize the nearest distances to infinity
    d_same = np.inf
    d_diff = np.inf
    
    #loop through the distances which just calculated
    for i in range(len(distances)):
        if same_class[i]:
            # check if it's the new nearest "same" distance
            d_same = min(d_same, distances[i])
        else:
            # check if it's the new nearest "different" distance
            d_diff = min(d_diff, distances[i])
    
#handle special cases
    # if nearest 'same' AND 'diff' are both 0, eg. duplicate points
    if d_same == 0 and d_diff == 0:
        return 0.0  # Convention: 0/0 = 0
    
    # if nearest 'same' is 0 (but 'diff' is not)
    elif d_same == 0:
        return np.inf  # a/0 = infinity for a > 0
    
    # if there were no 'same' neighbors found
    elif d_same == np.inf:
        return 0.0
    
    # if there were no 'different' neighbors found
    elif d_diff == np.inf:
        return np.inf 
    
    else:
        return d_diff / d_same #normal case