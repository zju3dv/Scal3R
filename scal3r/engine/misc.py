import warnings
from collections import abc
from importlib import import_module


def is_str(value) -> bool:
    return isinstance(value, str)


def is_seq_of(seq, expected_type, seq_type=None) -> bool:
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    return all(isinstance(item, expected_type) for item in seq)


def is_list_of(seq, expected_type) -> bool:
    return is_seq_of(seq, expected_type, seq_type=list)


def import_modules_from_strings(imports, allow_failed_imports: bool = False):
    if not imports:
        return None

    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]

    if not isinstance(imports, list):
        raise TypeError(f"imports must be a list, got {type(imports)}")

    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(f"{imp} is of type {type(imp)} and cannot be imported")
        try:
            imported_module = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f"{imp} failed to import and is ignored.", UserWarning)
                imported_module = None
            else:
                raise
        imported.append(imported_module)

    if single_import:
        return imported[0]
    return imported
