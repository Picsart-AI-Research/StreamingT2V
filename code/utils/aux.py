from types import FunctionType


def ensure_annotation_class(f: FunctionType):

    def wrapper(*args, **kwargs):
        keys = tuple(f.__annotations__.keys())
        args_converted = ()
        for ar in enumerate(args):
            expected_class = f.__annotations__.get(keys[ar[0]])
            if not isinstance(ar[1], expected_class):
                args_converted += (expected_class(ar[1]),)
            else:
                args_converted += (ar[1],)

        kwargs_ensured_class = {}
        for k, v in kwargs.items():
            expected_class = f.__annotations__.get(k)
            if not isinstance(v, expected_class):
                v = expected_class(v)
            kwargs_ensured_class[k] = v

        return f(*args_converted, **kwargs_ensured_class)

    return wrapper
