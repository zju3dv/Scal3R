import math
import logging
from time import strftime
from functools import wraps
from tqdm.auto import tqdm as auto_tqdm
from typing import Any, Callable, Iterable


try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.text import Text
    from rich.traceback import install as install_rich_traceback
    from rich.progress import Progress
    from rich.progress import BarColumn
    from rich.progress import filesize
    from rich.progress import ProgressColumn
    from rich.progress import TimeElapsedColumn
    from rich.progress import TimeRemainingColumn
    from tqdm.rich import FractionColumn
    from tqdm.std import tqdm as std_tqdm

    PROGRESS_CONSOLE_WIDTH = 120

    console = Console(
        soft_wrap=True,
        tab_size=4,
        width=None,
        log_time=True,
        log_path=False,
        log_time_format="%H:%M:%S",
    )
    progress_console = Console(
        soft_wrap=True,
        tab_size=4,
        width=PROGRESS_CONSOLE_WIDTH,
        log_time=True,
        log_path=False,
        log_time_format="%H:%M:%S",
    )
    install_rich_traceback(console=console)
    _USE_RICH = True
except ImportError:
    console = None
    progress_console = None
    FractionColumn = None
    Progress = None
    ProgressColumn = object
    std_tqdm = None
    _USE_RICH = False


def log(*parts: Any):
    message = " ".join(str(part) for part in parts)
    get_logger("scal3r").info(message)


def log_exceptions(logger: logging.Logger, message: str):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                logger.exception(message)
                raise

        return wrapper

    return decorator


def log_block(header: str, lines: Iterable[str], closing: str = ""):
    timestamp = strftime("%H:%M:%S")
    if _USE_RICH and console is not None:
        console.print(f"{timestamp} {header}", markup=False)
        for line in lines:
            console.print(f"    {line}", markup=False)
        if closing:
            console.print(closing, markup=False)
        return

    print(f"{timestamp} {header}")
    for line in lines:
        print(f"    {line}")
    if closing:
        print(closing)


def get_logger(name: str = "scal3r") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    if _USE_RICH and console is not None:
        handler: logging.Handler = RichHandler(
            console=console,
            show_time=True,
            show_level=False,
            show_path=False,
            markup=False,
            rich_tracebacks=True,
            log_time_format="%H:%M:%S",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


if _USE_RICH and progress_console is not None and std_tqdm is not None and FractionColumn is not None and Progress is not None:
    progress = Progress(console=progress_console, expand=False)


    class RateColumn(ProgressColumn):
        def __init__(self, unit: str = "", unit_scale: bool = False, unit_divisor: int = 1000, **kwargs):
            self.unit = unit
            self.unit_scale = unit_scale
            self.unit_divisor = unit_divisor
            super().__init__(**kwargs)

        def render(self, task):
            speed = task.speed
            if speed is None:
                return Text(f"? {self.unit}/s", style="progress.data.speed")
            if self.unit_scale:
                unit, suffix = filesize.pick_unit_and_suffix(
                    speed,
                    ["", "K", "M", "G", "T", "P", "E", "Z", "Y"],
                    self.unit_divisor,
                )
            else:
                unit, suffix = filesize.pick_unit_and_suffix(speed, [""], 1)
            ratio = speed / unit
            precision = max(0, 3 - int(math.log10(ratio))) if ratio > 0 else 0
            return Text(
                f"{ratio:,.{precision}f} {suffix}{self.unit}/s",
                style="progress.data.speed",
            )


    class TimeColumn(ProgressColumn):
        def render(self, task):
            log_time = progress_console.get_datetime()
            return Text(log_time.strftime("%H:%M:%S"), style="log.time")


    class tqdm_rich(std_tqdm):
        def __init__(self, *args, **kwargs):
            custom_progress = kwargs.pop("progress", None)
            options = kwargs.pop("options", {}).copy()
            super().__init__(*args, **kwargs)
            if self.disable:
                return

            d = self.format_dict
            if custom_progress is None:
                custom_progress = progress
                custom_progress.columns = (
                    TimeColumn(),
                    "[progress.description]{task.description}[progress.percentage]{task.percentage:>4.0f}%",
                    BarColumn(),
                    FractionColumn(
                        unit_scale=d["unit_scale"],
                        unit_divisor=d["unit_divisor"],
                    ),
                    TimeElapsedColumn(),
                    "<",
                    TimeRemainingColumn(),
                    RateColumn(
                        unit=d["unit"],
                        unit_scale=d["unit_scale"],
                        unit_divisor=d["unit_divisor"],
                    ),
                )
            elif options:
                for key, value in options.items():
                    setattr(custom_progress, key, value)

            self._prog = custom_progress
            self._prog.start()
            self._task_id = self._prog.add_task(self.desc or "", **d)

        def close(self):
            if self.disable:
                return
            self.disable = True
            self.display(refresh=True)
            if not hasattr(self, "_prog"):
                return
            if self._prog.finished:
                self._prog.stop()
                for task_id in list(self._prog.task_ids):
                    self._prog.remove_task(task_id)

        def clear(self, *_, **__):
            pass

        def display(self, refresh: bool = True, *_, **__):
            if not hasattr(self, "_prog"):
                return
            if self._task_id not in self._prog.task_ids:
                return
            self._prog.update(
                self._task_id,
                completed=self.n,
                description=self.desc,
                refresh=refresh,
            )

        def reset(self, total=None):
            if hasattr(self, "_prog") and self._task_id in self._prog.task_ids:
                self._prog.reset(self._task_id, total=total)
            super().reset(total=total)


    tqdm = tqdm_rich
else:
    def tqdm(*args: Any, **kwargs: Any):
        kwargs.setdefault("dynamic_ncols", False)
        return auto_tqdm(*args, **kwargs)
