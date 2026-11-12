"""ctypes bindings to the macOS libraries metalmeter reads from.

Every module imports on any platform. Off macOS, or when a library or symbol is missing,
the bound functions are None and the helpers return None or empty results.
"""
