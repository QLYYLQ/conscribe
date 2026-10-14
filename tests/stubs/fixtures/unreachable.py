"""Fixture: a class that no module binds to a name.

``HIDDEN[0]`` claims to live in this module (``__module__``) but the module
has no attribute naming it, so no stub can import it from anywhere.
"""

HIDDEN = [type("Hidden", (), {})]
