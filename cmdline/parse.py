

def parse_kwargs(parser):
    args, extra = parser.parse_known_args()
    kwargs = vars(args)
    kwargs.update(parse_extra(extra))
    kwargs = {key: try_cast(value) for key, value in kwargs.items()}
    return kwargs


def parse_extra(args):
    assert len(args) % 2 == 0
    kwargs = {}
    for key, value in zip(args[::2], args[1::2]):
        assert key.startswith('--')
        assert not value.startswith('--')
        key = key[2:]
        assert key not in kwargs
        kwargs[key] = value
    return kwargs


def try_cast(string: str):
    if not isinstance(string, str):
        return string

    for func in [strtonum, strtobool]:
        try:
            return func(string)
        except ValueError:
            pass
    return string


def strtonum(string):
    f = float(string)
    i = int(f)
    return f if (f != i) else i


def strtobool(string):
    lower = string.lower()
    if lower == 'true':
        return True
    if lower == 'false':
        return False
    raise ValueError
