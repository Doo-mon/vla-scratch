from typing import TypeVar, Generic, Sequence, SupportsIndex
from .transforms import TransformFn

T_co = TypeVar("T_co", covariant=True)

class Dataset(Generic[T_co]):
    """Base class for datasets."""

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[TransformFn]):
        self._dataset = dataset
        self._transforms = transforms

    def __getitem__(self, index: SupportsIndex) -> T_co:
        item = self._dataset[index]
        for transform in self._transforms:
            item = transform.compute(item)
        return item

    def __len__(self) -> int:
        return len(self._dataset)
