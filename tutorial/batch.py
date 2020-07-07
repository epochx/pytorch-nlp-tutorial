import torch
import numpy as np

class Batch(dict):
    def __init__(self, *args, **kwargs):
        super(Batch, self).__init__(*args, **kwargs)
        self.__dict__ = self
        self._is_torch = False

    def to_torch_(self, device):
        self._is_torch = False
        for key in self.keys():
            value = self[key]
            if isinstance(value, BatchTuple):
                value.to_torch_(device)
            if isinstance(value, np.ndarray):
                self[key] = torch.from_numpy(value).to(device)


class BatchTuple(object):
    def __init__(self, sequences, lengths, sublengths, masks):
        self.sequences = sequences
        self.lengths = lengths
        self.sublengths = sublengths
        self.masks = masks
        self._is_torch = False

    def to_torch_(self, device):
        if not self._is_torch:
            self.sequences = torch.tensor(
                self.sequences, device=device, dtype=torch.long
            )

            if self.lengths is not None:
                self.lengths = torch.tensor(
                    self.lengths, device=device, dtype=torch.long
                )

            if self.sublengths is not None:
                self.sublengths = torch.tensor(
                    self.sublengths, device=device, dtype=torch.long
                )
            if self.masks is not None:
                self.masks = torch.tensor(
                    self.masks, device=device, dtype=torch.float
                )

