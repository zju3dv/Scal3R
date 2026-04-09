from copy import deepcopy
from typing import Any, Mapping


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__()
        if args or kwargs:
            self.update(*args, **kwargs)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        del self[key]

    def update(self, data: Mapping[str, Any] | None = None, **kwargs) -> "DotDict":
        if data is None:
            data = {}
        elif not isinstance(data, Mapping):
            return super().update(data, **kwargs)

        merged = dict(data)
        merged.update(kwargs)
        for key, value in merged.items():
            if key in self and isinstance(self[key], DotDict) and isinstance(value, Mapping):
                self[key].update(value)
            else:
                self[key] = to_dot_dict(value)
        return self

    def copy(self) -> "DotDict":
        return DotDict(self)

    def to_dict(self) -> dict[str, Any]:
        return {key: to_plain_dict(value) for key, value in self.items()}


def to_dot_dict(value: Any) -> Any:
    if isinstance(value, DotDict):
        return value
    if isinstance(value, dict):
        return DotDict({key: to_dot_dict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [to_dot_dict(item) for item in value]
    return value


def to_plain_dict(value: Any) -> Any:
    if isinstance(value, DotDict):
        value = dict(value)
    if isinstance(value, dict):
        return {key: to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_dict(item) for item in value]
    return deepcopy(value)


dotdict = DotDict
