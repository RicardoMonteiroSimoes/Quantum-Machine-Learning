from math import pi

# General Helper Funtions


def mytestfunc() -> None:
    """
    Function testing
    """

    print("testfunc, and by the way:\nhere is pi:", pi)


def parity(x: str, n_classes: int) -> int:
    """
    Parity Function
    Counts occurences of 1 in binary input string

    :param string x: String in binary format (e.g. '101')
    :param int n_classes: Number of classes to calculate parity
    """

    return '{:b}'.format(x).count('1') % n_classes
