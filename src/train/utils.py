from itertools import cycle

def zip_cycle(*iterables):
    """
    Zip multiple iterables together, cycling through shorter iterables to match the length of the longest.

    This function is similar to the built-in zip() function, but instead of stopping when the shortest
    iterable is exhausted, it cycles through the shorter iterables to match the length of the longest one.

    Parameters:
    :param *iterables: Iterable
        Two or more iterables to be zipped together. Each iterable should support the len() function.

    :yield: tuple
        A tuple containing one item from each input iterable. When a shorter iterable is exhausted,
        it starts again from its beginning.
    """
    cycled_iterators = [cycle(iter(it)) for it in iterables]
    lengths = [len(it) for it in iterables]
    max_length = max(lengths)
    
    for _ in range(max_length):
        yield tuple(next(it) for it in cycled_iterators)
