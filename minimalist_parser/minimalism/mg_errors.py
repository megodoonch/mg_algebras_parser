class MGError(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, message=None):
        self.message = message


class SpICViolation(MGError):
    """Exception raised for specifier island constraint violations
    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message=None):
        self.message = message


class SMCViolation(MGError):
    """Exception raised for shortest move constraint violations
    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message=None):
        self.message = message


class ExistenceError(MGError):
    """Exception raised for errors in the input.
    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message

