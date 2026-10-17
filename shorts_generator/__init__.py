__all__ = ["generate_shorts"]


def __getattr__(name):
    # Pure transcript helpers can be used without loading API/media dependencies.
    if name == "generate_shorts":
        from .pipeline import generate_shorts

        globals()[name] = generate_shorts
        return generate_shorts
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
