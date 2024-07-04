from torch.profiler import record_function

def tracing_decorator(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with record_function(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator