import json
import os
from pathlib import Path
from json import JSONEncoder
from typing import Union
from dataclasses import dataclass, asdict, fields

import numpy as np
import torch


# File Management
# ---------------

def create_folder_if_not_exists(folder: str, raise_if_exists: bool = False):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    else:
        if raise_if_exists:
            raise ValueError(f"Folder '{folder}' already exists !")


class SafeOpen(object):
    def __init__(
        self,
        filepath: str,
        open_mode,
        raise_if_folder_exists: bool = False,
        overwrite: bool = False,
    ):
        self.filepath = Path(filepath)
        self.open_mode = open_mode
        self.raise_if_folder_exists = raise_if_folder_exists
        self.overwrite = overwrite

    def __enter__(self):
        is_writing = self.open_mode in ("w", "wb")

        # Create folder if it does not exist when writing
        if is_writing:
            create_folder_if_not_exists(folder=self.filepath.parent, raise_if_exists=self.raise_if_folder_exists)

        # Check if file already exists when writing
        if is_writing and (not self.overwrite) and os.path.isfile(self.filepath):
            raise FileExistsError(
                f"File '{self.filepath}' already exists... Please enable overwriting if you wish to proceed"
            )

        # Open file
        self.file = open(self.filepath, self.open_mode)
        return self.file

    def __exit__(self, *args):
        self.file.close()


# JSON
# ----

def _ndarray_encoding(array: Union[np.ndarray, torch.Tensor]) -> dict:
    # Handle tensorflow Tensors
    from_tensor = False
    if isinstance(array, torch.Tensor):
        array = array.numpy()
        from_tensor = True

    # Handle complex arrays
    is_complex = False
    array_data = dict(__ndarray__=array.real.tolist())
    if np.iscomplexobj(array):
        is_complex = True
        array_data["__ndarray_imag__"] = array.imag.tolist()

    return dict(
        **array_data,
        dtype=str(array.real.dtype),  # Keep only real part dtype
        to_tensor=from_tensor,
        is_complex=is_complex
    )


class JSONCustomEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.complexfloating):
            return complex(obj)
        elif isinstance(obj, (np.ndarray, torch.Tensor)):
            return _ndarray_encoding(obj)
        return JSONEncoder.default(self, obj)


def json_array_obj_hook(dct):
    if isinstance(dct, dict) and '__ndarray__' in dct:
        array = np.array(dct['__ndarray__'], dtype=dct['dtype'])
        if dct['is_complex']:
            array_imag = np.array(dct['__ndarray_imag__'], dtype=dct['dtype'])
            array = array + (1j * array_imag)
        if dct['to_tensor']:
            array = torch.from_numpy(array)
        return array
    return dct
