"""A Yelp-powered Restaurant Recommendation Program"""
# COMPLETE QUESTIONS 3-8

from abstractions import *
from data import ALL_RESTAURANTS, CATEGORIES, USER_FILES, load_user_file
from ucb import main, trace, interact
from utils import distance, mean, zip, enumerate, sample
from visualize import draw_map

##################################
# Phase 2: Unsupervised Learning #
##################################

# BEFORE CODING Q3-4, WATCH & TAKE NOTES ON:
# "PROJ 02 Phases 1 & 2 Videos: Key terms, k-means, & Q2-4" 
# (Google Classroom > Projects)

def find_closest(location, centroids):
    """Return the centroid in centroids that is closest to location.
    If multiple centroids are equally close, return the first one.

    >>> find_closest([3.0, 4.0], [[0.0, 0.0], [4.0, 3.0], [5.0, 5.0], [2.0, 3.0]])
    [2.0, 3.0]
    # Driver Name: Sandra Ximen
    # Navigator Name: Kaashvi Mittal
    # Uses the distance function to check which centroid in the list
    is closest to the given location. 
    # Returns the centroid that is closest to the location

    """
    # BEGIN Question 3
    dist_lst = [[centroid, distance(location, centroid)] for centroid in centroids]
    return min(dist_lst, key = lambda x: x[1])[0]
    # END Question 3

# print(find_closest([3.0, 4.0], [[0.0, 0.0], [4.0, 3.0], [5.0, 5.0], [2.0, 3.0]]))


def group_by_first(pairs):
    """Return a list of lists that relates each unique key in the [key, value]
    pairs to a list of all values that appear paired with that key.

    Arguments:
    pairs -- a sequence of pairs

    >>> example = [ [1, 2], [3, 2], [2, 4], [1, 3], [3, 1], [1, 2] ]
    >>> group_by_first(example)  # Values from pairs that start with 1, 3, and 2 respectively
    [[2, 3, 2], [2, 1], [4]]
    """
    # GIVEN - USE IN GROUP_BY_CENTROID BELOW
    keys = []
    for key, _ in pairs:
        if key not in keys:
            keys.append(key)
    return [[y for x, y in pairs if x == key] for key in keys]


def group_by_centroid(restaurants, centroids):
    """Return a list of clusters, where each cluster contains all restaurants
    nearest to a corresponding centroid in centroids. Each item in
    restaurants should appear once in the result, along with the other
    restaurants closest to the same centroid.
    """
    # The group_by_centroid function first iterates through each element in restaurants using a list comprehension.
    # Then, it finds the closest centroid to each element and creates sublists with each restaurant and corresponding centroid.
    # Finally, this function groups the restaurants with the same centroid together.

    # BEGIN Question 4
    # restaurants = sequence of restaurants, centroids = sequence of centroids 
    # list of clusters --> [[r1,c1], [r2,c3]]
    goal1 = [[find_closest(restaurant_location(elem), centroids), elem] for elem in restaurants]
    return group_by_first(goal1)
    # END Question 4

def find_centroid(cluster):
    """Return the centroid of the locations of the restaurants in cluster."""
    # This function finds the centroid of a cluster (which is a list of restaurants). 
    # location_list is a list of restaurant locations in the cluster. latitudes_list and longitudes_list extract the specific latitude and longitude from each restaurant.
    # Then, the mean is computed for both the latitudes and longitudes.
    # The average latitude and longitude, which is the centroid location, is returned as a list. 

    # BEGIN Question 5
    # restaurant abstraction 
    location_list = [restaurant_location(r) for r in cluster]
    latitudes_list = [elem[0] for elem in location_list]
    longitudes_list = [elem[1] for elem in location_list]
    latitudes_average = mean(latitudes_list)
    longitudes_average = mean(longitudes_list)

    return [latitudes_average, longitudes_average]
    # END Question 5


def k_means(restaurants, k, max_updates=100):
    """Use k-means to group restaurants by location into k clusters."""
    assert len(restaurants) >= k, 'Not enough restaurants to cluster'
    old_centroids, n = [], 0

    # Select initial centroids randomly by choosing k different restaurants
    centroids = [restaurant_location(r) for r in sample(restaurants, k)]

    while old_centroids != centroids and n < max_updates:
        old_centroids = centroids
    # The k-means function takes in a list of restaurants, the value k which represents the number of clusters, and a max updates threshold.
    # The first step involves randomly initializing a list of centroids. 
    # The k-means algorithm then iterates between two steps: 
        # First, group_by_centroid is called to group the restaurants into clusters. This list of lists is stored in the variable group_restaurants.
        # Then, for each element (which represents a cluster) in group_restaurants, find_centroid is called to compute the new centroid location.
    # Once, the centroid locations are optimized or the max update threshold has been reached, the function returns the list of centroids (which are each lists as well).

        # BEGIN Question 6
        group_restaurants = group_by_centroid(restaurants, centroids) # returns a list of clusters [[r1, r2, r4], [r3, r5]]
        centroids = [find_centroid(elem) for elem in group_restaurants] #update CURRENT centroids (centroids variable)
        # END Question 6
        n += 1
    return centroids


################################
# Phase 3: Supervised Learning #
################################

def find_predictor(user, restaurants, feature_fn):
    """Return a rating predictor (a function from restaurants to ratings),
    for a user by performing least-squares linear regression using feature_fn
    on the items in restaurants. Also, return the R^2 value of this model.

    Arguments:
    user -- A user
    restaurants -- A sequence of restaurants
    feature_fn -- A function that takes a restaurant and returns a number
    """
    xs = [feature_fn(r) for r in restaurants]
    ys = [user_rating(user, restaurant_name(r)) for r in restaurants]
  
    # This function takes in a user, a list of restaurants, and a feature function. 
    # The function returns the predictor function, which predicts the user's rating (ys) based on the features (xs). 
    # It also returns r_squared which is an indicator of how well the predictor function fits the data.

    # BEGIN Question 7
    S_xx = sum((elem - mean(xs))**2 for elem in xs)
    S_yy = sum((elem - mean(ys))**2 for elem in ys)
    S_xy = sum((x - mean(xs))*(y - mean(ys)) for x,y in zip(xs, ys))
    
    # 3-line implementation allowed because autograder works
    # returns the predictor function, which predicts the user's rating
    b = S_xy/S_xx
    a = mean(ys) - b * mean(xs)
    r_squared = S_xy**2 / (S_xx * S_yy)
    # END Question 7

    def predictor(restaurant):
        return b * feature_fn(restaurant) + a

    return predictor, r_squared


def best_predictor(user, restaurants, feature_fns):
    """Find the feature within feature_fns that gives the highest R^2 value
    for predicting ratings by the user; return a predictor using that feature.

    Arguments:
    user -- A user
    restaurants -- A list of restaurants
    feature_fns -- A sequence of functions that each takes a restaurant
    """
    reviewed = user_reviewed_restaurants(user, restaurants)
    # Uses a list comprehension to iterate through each features in feature_fn and applies the function find_predictor 
    # then, we use the max function to find the highest R^2 value, and then returns the predictor function with the highest R^2 value   

    # BEGIN Question 8
    return max([find_predictor(user,reviewed,ff) for ff in feature_fns], key=lambda z: z[1])[0]
    # END Question 8


def rate_all(user, restaurants, feature_fns):
    """Return the predicted ratings of restaurants by user using the best
    predictor based on a function from feature_fns.

    Arguments:
    user -- A user
    restaurants -- A list of restaurants
    feature_fns -- A sequence of feature functions
    """
    predictor = best_predictor(user, ALL_RESTAURANTS, feature_fns)
    reviewed = user_reviewed_restaurants(user, restaurants)
    # BEGIN Question 9
    "*** YOUR CODE HERE ***"
    # END Question 9

# QUESTION 10 IS NOT REQUIRED. OPTIONAL CHALLENGE.
def search(query, restaurants):
    """Return each restaurant in restaurants that has query as a category.

    Arguments:
    query -- A string
    restaurants -- A sequence of restaurants
    """
    # BEGIN Question 10
    "*** YOUR CODE HERE ***"
    # END Question 10


def feature_set():
    """Return a sequence of feature functions."""
    return [lambda r: mean(restaurant_ratings(r)),
            restaurant_price,
            lambda r: len(restaurant_ratings(r)),
            lambda r: restaurant_location(r)[0],
            lambda r: restaurant_location(r)[1]]


@main
def main(*args):
    import argparse
    parser = argparse.ArgumentParser(
        description='Run Recommendations',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-u', '--user', type=str, choices=USER_FILES,
                        default='test_user',
                        metavar='USER',
                        help='user file, e.g.\n' +
                        '{{{}}}'.format(','.join(sample(USER_FILES, 3))))
    parser.add_argument('-k', '--k', type=int, help='for k-means')
    parser.add_argument('-q', '--query', choices=CATEGORIES,
                        metavar='QUERY',
                        help='search for restaurants by category e.g.\n'
                        '{{{}}}'.format(','.join(sample(CATEGORIES, 3))))
    parser.add_argument('-p', '--predict', action='store_true',
                        help='predict ratings for all restaurants')
    parser.add_argument('-r', '--restaurants', action='store_true',
                        help='outputs a list of restaurant names')
    args = parser.parse_args()

    # Output a list of restaurant names
    if args.restaurants:
        print('Restaurant names:')
        for restaurant in sorted(ALL_RESTAURANTS, key=restaurant_name):
            print(repr(restaurant_name(restaurant)))
        exit(0)

    # Select restaurants using a category query
    if args.query:
        restaurants = search(args.query, ALL_RESTAURANTS)
    else:
        restaurants = ALL_RESTAURANTS

    # Load a user
    assert args.user, 'A --user is required to draw a map'
    user = load_user_file('{}.dat'.format(args.user))

    # Collect ratings
    if args.predict:
        ratings = rate_all(user, restaurants, feature_set())
    else:
        restaurants = user_reviewed_restaurants(user, restaurants)
        names = [restaurant_name(r) for r in restaurants]
        ratings = {name: user_rating(user, name) for name in names}

    # Draw the visualization
    if args.k:
        centroids = k_means(restaurants, min(args.k, len(restaurants)))
    else:
        centroids = [restaurant_location(r) for r in restaurants]
    draw_map(centroids, restaurants, ratings)
