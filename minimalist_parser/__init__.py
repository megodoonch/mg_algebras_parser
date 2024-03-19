import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.WARNING, format='%(levelname)s (%(name)s) - %(message)s')

# root_logging = logging.getLogger(__name__)
# root_logging.setLevel(logging.DEBUG)  # this seems to set the minimum logging level. Set what you actually want below.
#
# # log to standard out
# stream_handler = logging.StreamHandler(sys.stdout)
# # choose how much you want to print to stout (DEBUG is everything, ERROR is almost nothing)
# stream_handler.setLevel(logging.DEBUG)  # log everything
# # stream_handler.setLevel(logging.INFO)  # only log info and higher
# # stream_handler.setLevel(logging.WARNING)
# # stream_handler.setLevel(logging.ERROR)
# formatter = logging.Formatter('%(levelname)s (%(name)s) - %(message)s')  # in minimalist_parser/__init__.py')
# stream_handler.setFormatter(formatter)
# root_logging.addHandler(stream_handler)
