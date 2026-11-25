class _Fn:
    argtypes = None
    def __call__(self, *a, **k):
        raise RuntimeError("wand stub: ImageMagick not installed")

class _Lib:
    def __getattr__(self, name):
        f = _Fn(); setattr(self, name, f); return f

library = _Lib()
