"""Utilities for Maps"""

from math import sqrt
from random import sample

# Rename the built-in zip (http://docs.python.org/3/library/functions.html#zip)
_zip = zip

def map_and_filter(s, map_fn, filter_fn):
    """Returns a new list containing the results of calling map_fn on each
    element of sequence s for which filter_fn returns a true value.

    >>> square = lambda x: x * x
    >>> is_odd = lambda x: x % 2 == 1
    >>> map_and_filter([1, 2, 3, 4, 5], square, is_odd)
    [1, 9, 25]
    """
    # This implementation uses a list comprehension to map_fn on each element in an inputted
    # sequence if the function filter_fn is True

    # BEGIN Question 0.1
    return [map_fn(elem) for elem in s if filter_fn(elem) == True]
    # END Question 0.1

def key_of_min_value(d):
    """Returns the key in a dict d that corresponds to the minimum value of d.

    >>> letters = {'a': 6, 'b': 5, 'c': 4, 'd': 5}
    >>> min(letters)
    'a'
    >>> key_of_min_value(letters)
    'c'
    """
    # This implementation uses the built-in min function and the key function.
    # The key function takes in each element of the dictionary and returns the corresponding value. 
    # The min function then returns the key with the minimum value. 

    # BEGIN Question 0.2
    return min(d, key=lambda x: d[x])
    # END Question 0.2

def zip(*sequences):
    """Returns a list of lists, where the i-th list contains the i-th
    element from each of the argument sequences.

    >>> zip(range(0, 3), range(3, 6))
    [[0, 3], [1, 4], [2, 5]]
    >>> for a, b in zip([1, 2, 3], [4, 5, 6]):
    ...     print(a, b)
    1 4
    2 5
    3 6
    >>> for triple in zip(['a', 'b', 'c'], [1, 2, 3], ['do', 're', 'mi']):
    ...     print(triple)
    ['a', 1, 'do']
    ['b', 2, 're']
    ['c', 3, 'mi']
    """
    # GIVEN - USE IN ENUMERATE FUNCTION BELOW
    return list(map(list, _zip(*sequences)))

def enumerate(s, start=0):
    """Returns a list of lists, where the i-th list contains i+start and
    the i-th element of s.

    >>> enumerate([6, 1, 'a'])
    [[0, 6], [1, 1], [2, 'a']]
    >>> enumerate('five', 5)
    [[5, 'f'], [6, 'i'], [7, 'v'], [8, 'e']]
    """
    # This function uses the zip function to put elements from two sequences together in lists so that each element from the first
    # sequence is place in a list with the corresponding element (by index) from the second sequence.
    # The first sequence is the range function which begins at the start and ends at the start + the length of the sequence s. 
    # The second sequence is s

    # BEGIN Question 0.3
    return list(zip(range(start, start + len(s)), s))
    # END Question 0.3

def distance(pos1, pos2):
    """Returns the Euclidean distance between pos1 and pos2, which are pairs.

    >>> distance([1, 2], [4, 6])
    5.0
    """
    return sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

def mean(s):
    """Returns the arithmetic mean of a sequence of numbers s.

    >>> mean([-1, 3])
    1.0
    >>> mean([0, -3, 2, -1])
    -0.5
    """
  
    # This function first uses the assert statement to ensure taht the length of the sequence s is greater than 0.
    # Then, the function uses a list comprehension to sum each element in the sequence.
    # Finally, the sum is divided by the length of the sequence, returning the mean.
    # BEGIN Question 1
    assert len(s) > 0
    return sum([elem for elem in s]) / len(s)
    
    # END Question 1
