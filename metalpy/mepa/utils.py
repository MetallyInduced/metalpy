import traceback


def traverse_args(args, kwargs, func):
    # 对所有的参数进行处理
    _args = []
    for arg in args:
        _args.append(func(arg))

    _kwargs = {}
    for k, v in kwargs.items():
        _kwargs[k] = func(v)

    return _args, _kwargs


def structured_traverse(struct, func):
    if isinstance(struct, dict):
        ret = {key: structured_traverse(element, func) for key, element in struct.items()}
    elif isinstance(struct, list):
        ret = [structured_traverse(element, func) for element in struct]
    else:
        ret = func(struct)

    return ret


def exception_caught(func, *args, **kwargs):
    """用于处理loky无法正确识别并捕获异常的问题
    TODO: 参考multiprocessing进行重新创建堆栈，从而帮助调试
    """
    try:
        return func(*args, **kwargs)
    except BaseException as e:
        traceback.print_exc()
        raise e


def is_main_thread_of_main_process():
    import threading
    import multiprocessing

    return (
        threading.current_thread() is threading.main_thread()
        and multiprocessing.parent_process() is None
    )


def is_serial():
    return is_main_thread_of_main_process()
