"""The repository's own methods - one module per method, discovered rather than listed.

Nothing in this package's ``__init__`` imports the method modules by name, and nothing outside it
does either. :func:`usis.methods.registry.load_builtin_methods` walks the package with
``pkgutil`` and imports what it finds, which is what makes brief §6.1's requirement literally true:
*adding a method must never require touching the harness.* Adding one here is adding one file.
"""
