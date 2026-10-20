"""Errors for unreadable or unsafe uploads. The API maps all of them to 400 INVALID_INPUT."""


class IngestError(ValueError):
    """The upload is unsafe, malformed or in an unsupported form. The message is safe to show the caller."""
