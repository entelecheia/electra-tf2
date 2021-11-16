import datetime
from contextlib import contextmanager
from timeit import default_timer

@contextmanager
def elapsed_timer(format_time=False):
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: str(datetime.timedelta(seconds=elapser())) if format_time else elapser()
    end = default_timer()
    elapser = lambda: end-start
    