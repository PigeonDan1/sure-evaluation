# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Request-scoped bounded number cache."""

from collections import OrderedDict
from collections.abc import Iterator, MutableMapping


class BoundedNumberCache(MutableMapping[str, str]):
    """Fixed-capacity LRU; eviction only triggers recompute and does not change normalization semantics."""

    def __init__(self, max_size: int = 4096):
        if max_size <= 0:
            raise ValueError("number cache max_size must be greater than 0")
        self.max_size = max_size
        self._values: OrderedDict[str, str] = OrderedDict()

    def __getitem__(self, key: str) -> str:
        value = self._values[key]
        self._values.move_to_end(key)
        return value

    def __setitem__(self, key: str, value: str) -> None:
        self._values[key] = value
        self._values.move_to_end(key)
        while len(self._values) > self.max_size:
            self._values.popitem(last=False)

    def __delitem__(self, key: str) -> None:
        del self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __contains__(self, key: object) -> bool:
        return key in self._values
