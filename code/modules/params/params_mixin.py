
class AsDictMixin:
    def to_dict(self):

        keys = [entry for entry in dir(self) if not callable(getattr(
            self, entry)) and not entry.startswith("__")]

        result_dict = {}
        for key in keys:
            result_dict[key] = getattr(self, key)
        return result_dict

    def __str__(self) -> str:
        return self.to_dict().__str__()