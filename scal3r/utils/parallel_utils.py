from typing import Callable, Dict, List
from multiprocessing.pool import Pool, ThreadPool

from scal3r.utils.console_utils import tqdm


def parallel_execution(
    *args,
    action: Callable,
    num_workers: int = 32,
    print_progress=False,
    sequential=False,
    async_return=False,
    desc=None,
    use_process=False,
    **kwargs,
):
    """
    Executes a given function in parallel using threads or processes.
    When using threads, the parallelism is achieved during IO blocking.
    """

    def get_length(args: List, kwargs: Dict):
        for a in args:
            if isinstance(a, list):
                return len(a)
        for v in kwargs.values():
            if isinstance(v, list):
                return len(v)
        raise NotImplementedError

    def get_action_args(length: int, args: List, kwargs: Dict, i: int):
        action_args = [(arg[i] if isinstance(arg, list) and len(arg) == length else arg) for arg in args]
        action_kwargs = {
            key: (kwargs[key][i] if isinstance(kwargs[key], list) and len(kwargs[key]) == length else kwargs[key])
            for key in kwargs
        }
        return action_args, action_kwargs

    if not sequential:
        if use_process:
            pool = Pool(processes=num_workers)
        else:
            pool = ThreadPool(processes=num_workers)

        results = []
        asyncs = []
        length = get_length(args, kwargs)
        for i in range(length):
            action_args, action_kwargs = get_action_args(length, args, kwargs, i)
            async_result = pool.apply_async(action, action_args, action_kwargs)
            asyncs.append(async_result)

        if not async_return:
            for async_result in tqdm(asyncs, desc=desc, disable=not print_progress):
                results.append(async_result.get())
            pool.close()
            pool.join()
            return results
        else:
            return pool
    else:
        results = []
        length = get_length(args, kwargs)
        for i in tqdm(range(length), desc=desc, disable=not print_progress):
            action_args, action_kwargs = get_action_args(length, args, kwargs, i)
            async_result = action(*action_args, **action_kwargs)
            results.append(async_result)
        return results
