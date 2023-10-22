from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Collection, Callable, TYPE_CHECKING

import imageio
import pyvista as pv
import tqdm

from metalpy.mepa import LinearExecutor, Executor, ParallelProgress
from metalpy.utils.dhash import dhash
from metalpy.utils.file import make_cache_file_path, PathLike
from metalpy.utils.model import convert_model_dict_to_multiblock, pv_ufunc_apply, DataSetLike
from metalpy.utils.time import Timer
from .dataset_wrapper import DataSetWrapper, TDataSetWrapper

if TYPE_CHECKING:
    from metalpy.carto.utils.crs import CRSLike

ModelPostprocessor = Callable[[DataSetLike, Path], DataSetLike]


class UniversalDataSet(DataSetWrapper):
    Transforms = {
        pv.DataSet.rotate_x.__name__,
        pv.DataSet.rotate_y.__name__,
        pv.DataSet.rotate_z.__name__,
        pv.DataSet.rotate_vector.__name__,
        pv.DataSet.translate.__name__,
        pv.DataSet.scale.__name__,
        pv.DataSet.flip_x.__name__,
        pv.DataSet.flip_y.__name__,
        pv.DataSet.flip_z.__name__,
        pv.DataSet.flip_normal.__name__,
    }

    def __init__(self, dataset: DataSetLike, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)

    def warp(
            self: TDataSetWrapper,
            src_crs: 'CRSLike',
            *,
            crs=None,
            query=None,
            inplace=False,
            return_crs=False
    ) -> TDataSetWrapper:
        """转换数据集的坐标系

        Parameters
        ----------
        src_crs
            源坐标系
        crs
            目标坐标系
        query
            按指定模式搜索坐标系
        inplace
            指定转换是否就地操作
        return_crs
            指示时候返回采用的新坐标系，常用与配合`query`，获取查询到的坐标系信息

        Returns
        -------
        result
            转换坐标系后的数据集，如果指定了`return_crs=True`，则额外返回转换后的坐标系信息
        """
        from metalpy.carto.pyvista.crs import warp_dataset

        return self.wrap(warp_dataset(
            self.dataset,
            src_crs,
            crs=crs,
            query=query,
            inplace=inplace,
            return_crs=return_crs
        ))

    @classmethod
    def read(
            cls: type[TDataSetWrapper],
            model_path: PathLike,
            callback: ModelPostprocessor | None = None
    ) -> pv.DataSet | TDataSetWrapper:
        model = _load_model(model_path, callback=callback)
        return cls(model)

    @classmethod
    def read_all(
            cls: type[TDataSetWrapper],
            path,
            progress=True,
            verbose=True,
            cache=True,
            preview: int | bool = False,
            callback: ModelPostprocessor | None = None,
            executor: Executor | None = None
    ) -> pv.DataSet | TDataSetWrapper:
        """从指定路径中读取全部模型文件

        Parameters
        ----------
        path
            模型路径根目录
        progress
            是否显示进度条
        verbose
            是否显示提示信息
        cache
            是否缓存较大的模型到本地缓存路径下
        preview
            指示只载入前若干个模型文件
        callback
            针对每个载入模型的回调处理函数
        executor
            指定用于执行载入任务的执行器

        Returns
        -------
        dataset
            载入的全部模型数据数据

        Notes
        -----
        使用ProcessExecutor等依赖pickle序列化回传结果的执行器时，序列化PyVista对象耗时会远大于读取时间，因此应用场景有限。

        TODO: 寻找使用ThreadExecutor进行加速的可能性。
            可能可以依靠取消GIL后的ThreadExecutor？
            当前多线程读取PyVista对象依然会受限于GIL，导致IO阻塞。
        """
        path = Path(path)
        model_name = path.name
        cache_path = make_cache_file_path(f'models/{dhash(path).hexdigest(6)}/{model_name}.vtm')

        if preview is True:
            preview = 10

        if cache and cache_path.exists():
            timer = Timer()
            with timer:
                ret = pv.read(cache_path)  # TODO: 判断`texture`选项？
            if verbose:
                print(f'Model loaded from `{path}` in {timer}.')
        else:
            model_files = [
                os.path.join(base, f)
                for base, _, files in os.walk(path)
                for f in files
                if _check_is_readable(Path(f))
            ]

            if preview:
                model_files = model_files[:preview]

            if progress is True:
                total = len(model_files)
                progress = executor.progress(total=total)
            elif isinstance(progress, tqdm.tqdm):
                pass
            else:
                progress = None

            if executor is None:
                executor = LinearExecutor()

            with Timer() as timer:
                allocator = executor.arrange(model_files)
                futures = executor.distribute(_load_models_from_directory, cwd=path, paths=allocator, callback=callback,
                                              progress=progress)

                model_dicts = executor.gather(futures)

                ret = pv.MultiBlock()
                for model_dict in model_dicts:
                    convert_model_dict_to_multiblock(model_dict, root=ret)

                if progress is not None:
                    progress.close()

                if cache and timer.elapsed > 5:
                    if verbose:
                        print(f'Saving loaded model to `{cache_path}`...')

                    with timer:
                        ret.save(cache_path, binary=True)
                        if verbose:
                            print(f'Model saved in {timer}.')

        return cls(ret)

    def __getattr__(self, item):
        if item in UniversalDataSet.Transforms:
            def wrapper(*args, **kwargs):
                inplace = kwargs.pop('inplace', False)
                new_dataset = pv_ufunc_apply(
                    self.dataset,
                    lambda dataset: getattr(dataset, item)(*args, **kwargs, inplace=True),
                    inplace=inplace
                )
                if inplace:
                    return self
                else:
                    return self.wrap(new_dataset)

            return wrapper
        else:
            return super().__getattr__(item)


__MODEL_FORMATS = set(pv.core.utilities.reader.CLASS_READERS.keys())
__MODEL_FORMATS.difference_update({
    *imageio.config.known_extensions
})


def _check_is_readable(file: Path, excludes: Collection = None):
    suffix = file.suffix
    flag = file.suffix in __MODEL_FORMATS
    if excludes is not None:
        flag &= suffix not in excludes
    return flag


def _load_models_from_directory(
        cwd,
        paths=None,
        callback=None,
        progress: ParallelProgress | None = None,
):
    root_model: dict[str, Any] = {}

    cwd = Path(cwd)
    if paths is None:
        paths = list(cwd.iterdir())

    for path in paths:
        name_parts, sub_model = _check_directory_and_load(
            cwd=cwd,
            child_path=Path(path),
            callback=callback,
            progress=progress
        )

        if sub_model is not None:
            node = root_model
            for part in name_parts[:-1]:
                node = node.setdefault(part, {})
            node[name_parts[-1]] = sub_model

    return root_model


def _load_model(model_path, callback=None) -> DataSetLike:
    model_path = Path(model_path)
    model = pv.read(os.fspath(model_path))

    if callback is not None:
        model = callback(model, model_path)

    return model


def _check_directory_and_load(
        cwd: Path,
        child_path: Path,
        callback=None,
        progress: ParallelProgress | None = None
):
    sub_model = None
    if _check_is_readable(child_path):
        try:
            sub_model = _load_model(child_path, callback=callback)
        except IOError:
            pass

        if progress is not None:
            progress.update(1)

    if sub_model is not None:
        return child_path.relative_to(cwd).parts, sub_model
    else:
        return tuple(), None
