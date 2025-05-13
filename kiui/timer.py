import os
import torch
from functools import wraps

class sync_timer:
    """Synchronized timer to count the inference time of `nn.Module.forward` or else.

    :class:`sync_timer` can be used as a context manager or a decorator.

    Example as context manager:
    
    .. code-block:: python

        with timer('name'):
            run()

    Example as decorator:

    .. code-block:: python

        @timer('name')
        def run():
            pass
        
    Args:
        name (str, optional): name of the timer. Defaults to None.
        flag_env (str, optional): environment variable to check if logging is enabled. Defaults to "TIMER".
        logger_func (Callable, optional): function to log the result. Defaults to ``print``.

    Note:
        Set environment variable ``$flag_env`` to ``1`` to enable logging! default is ``TIMER=1``.
    """

    def __init__(self, name=None, flag_env="TIMER", logger_func=print):
        self.name = name
        self.flag_env = flag_env
        self.logger_func = logger_func

    def __enter__(self):
        if os.environ.get(self.flag_env, "0") == "1":
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()

    def __exit__(self, exc_type, exc_value, exc_tb):
        if os.environ.get(self.flag_env, "0") == "1":
            self.end.record()
            torch.cuda.synchronize()
            delta_time = self.start.elapsed_time(self.end)
            if self.name is not None:
                self.logger_func(f"{self.name} takes {delta_time/1000:.3f} s")

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                result = func(*args, **kwargs)
            return result

        return wrapper
