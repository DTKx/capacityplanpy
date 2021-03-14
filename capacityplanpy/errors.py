class Error(Exception):
    """Base class for exceptions in this module."""

    pass


class CountError(Error):
    """Exception raised for errors in the expected number of values.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class InvalidValuesError(Error):
    """Exception raised for errors in the expected values.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message
