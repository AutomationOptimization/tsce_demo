from contextlib import contextmanager

@contextmanager
def freeze_time(*args, **kwargs):
    yield
