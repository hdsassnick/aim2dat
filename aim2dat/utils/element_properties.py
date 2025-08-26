"""Deprecated module to retrieve physical and chemical properties of elements."""

# Internal library imports
import aim2dat.elements as new_elements


def get_atomic_radius(element, radius_type="covalent"):
    """
    Return the covalent or van der Waals radius of the element. The following sources are
    used for different radius types:

    * ``'covalent'`` are from :doi:`10.1039/B801115J`.
    * ``'vdw'`` are from :doi:`10.1039/C3DT50599E`.
    * ``'chen_manz'`` are from :doi:`10.1039/C9RA07327B`.
    * ``'vdw_charry_tkatchenko'`` are from :doi:`10.26434/chemrxiv-2024-m3rtp-v2`.

    Parameters
    ----------
    element : str or int
        Atomic number, name or symbol of the element.
    radius_type : str (optional)
        Radius type. Valid options are ``'covalent'``, ``'vdw'``, ``'chen_manz'``,
        or ``'vdw_charry_tkatchenko'``.

    Returns
    -------
    radius : float
        Atomic radius of the element.

    Raises
    ------
    ValueError
        If ``radius_type`` is not supported or has the wrong format.
    """
    from warnings import warn

    warn(
        "This function will be removed, "
        + "please use the functions implemented in `aim2dat.elements` instead.",
        DeprecationWarning,
        2,
    )
    return new_elements.get_atomic_radius(element, radius_type=radius_type)


def get_electronegativity(element, scale="pauling"):
    """
    Return the electronegativity of the element.

    Parameters
    ----------
    element : str or int
        Atomic number, name or symbol of the element.
    scale : str (optional)
        Electronegativity scale. Supported values are ``'pauling'`` and ``'allen'``.

    Returns
    -------
    float or None
        Electronegativity of the element.
    """
    from warnings import warn

    warn(
        "This function will be removed, "
        + "please use the functions implemented in `aim2dat.elements` instead.",
        DeprecationWarning,
        2,
    )
    return new_elements.get_electronegativity(element, scale=scale)


def get_atomic_number(element):
    """
    Return atomic number of the element from element symbol or name.

    Parameters
    ----------
    element : str or int
        Atomic number, name or symbol of the element.

    Returns
    -------
    int
        Atomic number of the element.
    """
    from warnings import warn

    warn(
        "This function will be removed, "
        + "please use the functions implemented in `aim2dat.elements` instead.",
        DeprecationWarning,
        2,
    )
    return new_elements.get_atomic_number(element)


def get_element_symbol(element):
    """
    Return symbol of the element from element number or name.

    Parameters
    ----------
    element : str or int
        Atomic number, name or symbol of the element.

    Returns
    -------
    str
        Symbol of the element.
    """
    from warnings import warn

    warn(
        "This function will be removed, "
        + "please use the functions implemented in `aim2dat.elements` instead.",
        DeprecationWarning,
        2,
    )
    return new_elements.get_element_symbol(element)


def get_atomic_mass(element):
    """
    Return atomic mass of the element from the atomic number, element symbol or name.

    Parameters
    ----------
    element : str or int
        Atomic number, name or symbol of the element.

    Returns
    -------
    int
        Atomic mass of the element.
    """
    from warnings import warn

    warn(
        "This function will be removed, "
        + "please use the functions implemented in `aim2dat.elements` instead.",
        DeprecationWarning,
        2,
    )
    return new_elements.get_atomic_mass(element)


def get_val_electrons(element):
    """
    Return number of valence electrons of the element from the atomic number, element symbol
    or name.

    Parameters
    ----------
    element : str or int
        Atomic number, name or symbol of the element.

    Returns
    -------
    int
        Number of valence electrons of the element.
    """
    from warnings import warn

    warn(
        "This function will be removed, "
        + "please use the functions implemented in `aim2dat.elements` instead.",
        DeprecationWarning,
        2,
    )
    return new_elements.get_val_electrons(element)


def get_element_groups(element):
    """
    Return groups that contain the element from the atomic number, element symbol or name.

    Parameters
    ----------
    element : str or int
        Atomic number, name or symbol of the element.

    Returns
    -------
    groups : set
        Set of groups.
    """
    from warnings import warn

    warn(
        "This function will be removed, "
        + "please use the functions implemented in `aim2dat.elements` instead.",
        DeprecationWarning,
        2,
    )
    return new_elements.get_element_groups(element)


def get_group(group_label):
    """
    Return all elements in the group.

    Parameters
    ----------
    group_label : str
        Group label.

    Returns
    -------
    elements : set
        Set of element symbols..
    """
    from warnings import warn

    warn(
        "This function will be removed, "
        + "please use the functions implemented in `aim2dat.elements` instead.",
        DeprecationWarning,
        2,
    )
    return new_elements.get_group(group_label)
