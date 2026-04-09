from os import PathLike
from os.path import abspath, dirname, expanduser, isabs, isfile, join


def get_release_root() -> str:
    return dirname(dirname(dirname(abspath(__file__))))


def get_data_root() -> str:
    return join(get_release_root(), "data")


def get_result_root() -> str:
    return join(get_data_root(), "result")


def get_checkpoint_root() -> str:
    return join(get_data_root(), "checkpoints")


def get_custom_result_root() -> str:
    return join(get_result_root(), "custom")


def get_benchmark_result_root() -> str:
    return join(get_result_root(), "evaluation")


def get_default_output_dir(name: str = "run") -> str:
    return join(get_custom_result_root(), name)


def get_default_evaluation_dir(name: str = "camera_metrics") -> str:
    return join(get_benchmark_result_root(), name)


def resolve_release_path(path_like: str | PathLike[str]) -> str:
    path = expanduser(str(path_like))
    if isabs(path):
        return abspath(path)
    return abspath(join(get_release_root(), path))


def check_file_exist(filename, msg_tmpl: str = 'file "{}" does not exist') -> None:
    if not isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))
