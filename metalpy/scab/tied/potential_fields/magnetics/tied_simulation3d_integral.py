from typing import Union, Iterable

import numpy as np
import taichi as ti
import tqdm

from metalpy.utils.taichi import (
    ti_size_max,
    ti_kernel,
    ti_cfg,
    ti_ndarray,
    ti_func,
    ti_data_oriented,
    ti_ndarray_from,
    ensure_contiguous
)
from metalpy.utils.taichi_kernels import ti_index
from metalpy.utils.type import ensure_as_iterable, not_none_or_default
from ...value_observer import ValueObserver


class Receiver:
    def __init__(self,
                 receiver_locations: np.ndarray,
                 components: Union[str, Iterable[str]]):
        self.receiver_locations = receiver_locations
        self.components = components

    @property
    def n_rows(self):
        return self.receiver_locations.shape[0] * len(self.components)


@ti_data_oriented
class TaichiSimulation3DIntegral:
    MType_Vector = 0
    MType_Scalar = 1
    NonAccumulative = 0
    Accumulative = 1
    Layout_SoA = ti.Layout.SOA
    Layout_AoS = ti.Layout.AOS

    def __init__(self,
                 receivers: Union[Receiver, list[Receiver]],
                 xn: np.ndarray,
                 yn: np.ndarray,
                 zn: np.ndarray,
                 base_cell_sizes: np.ndarray,
                 model_type: int = MType_Scalar,
                 row_stype: int = Layout_SoA,
                 col_stype: int = Layout_AoS,
                 tmi_projection: np.ndarray = None,
                 magnetization: np.ndarray = None,):
        """用于计算磁场的积分正演

        Parameters
        ----------
        receivers
            接收点信息包含坐标和待求解分量，
            坐标应为array(n_obs, 3)，
            待求解分量可选 'tmi', 'bx', 'by', 'bz', 'bxx', 'bxy', 'bxz', 'byy', 'byz', 'bzz'
        xn, yn, zn
            网格边界坐标，均为array(n_cells, 2)
        base_cell_sizes
            基础单元大小，array(3,)
        model_type
            模型类型，可选MType_Scalar或MType_Vector
        row_stype
            核矩阵行存储类型/正演算子的输入类型，可选Layout_SoA或Layout_AoS
        col_stype
            核矩阵列存储类型/正演算子的输出类型，可选Layout_SoA或Layout_AoS
        tmi_projection
            tmi投影矩阵，array(3,)，在不需要计算tmi分量时可不传
        magnetization
            磁化矩阵，array(n_cells, 3)，默认为全1
        """
        receivers: list[Receiver] = ensure_as_iterable(receivers)

        if tmi_projection is None:
            for receiver in receivers:
                assert 'tmi' not in receiver.components, "'tmi_projection' is required when computing tmi"
            tmi_projection = np.asarray([0])  # dummy variable

        if magnetization is None:
            magnetization = 1

        magnetization = np.asarray(magnetization)
        if magnetization.ndim == 2 and np.allclose(magnetization - magnetization[0], 0):
            # 标量模式下SimPEG会将地磁场重复n_cell行，判断如果重复则可以压缩磁化矩阵
            magnetization = magnetization[0]

        self.receivers = receivers
        self.xn = xn
        self.yn = yn
        self.zn = zn
        self.n_cells = xn.shape[0]
        self.base_cell_sizes = base_cell_sizes
        self.magnetization = magnetization
        self.tmi_projection = tmi_projection
        self.row_stype = row_stype
        self.col_stype = col_stype
        self.model_type = model_type

    def dpred(self,
              model: np.ndarray = None,
              progress: tqdm.tqdm = None):
        """计算正演结果

        Parameters
        ----------
        model
            模型参数
        progress
            tqdm进度条对象

        Returns
        -------
        ret
            正演结果

        Notes
        -----
            如果model_type==MType_Scalar，则model应为array(n_cells,)

            如果model_type==MType_Vector，则model应为array(n_cells, 3)

            如果model为None，则返回核矩阵。如果为scalar模型，核矩阵列数为n_cells，如果为vector模型，核矩阵列数为3*n_cells

            如果model不为None，则返回正演结果列向量
        """
        is_cpu = ti_cfg().arch == ti.cpu

        forward_only = model is not None
        model = not_none_or_default(model, supplier=lambda: np.empty(3, dtype=np.int8))

        if self.model_type == self.MType_Scalar:
            n_cols = self.n_cells
        else:
            n_cols = self.n_cells * 3
            model = model.reshape(-1, 3)

        if forward_only:
            n_cols = 1

        n_rows = sum(receiver.n_rows for receiver in self.receivers)
        if is_cpu:
            ret = np.zeros((n_rows, n_cols), dtype=np.float64)
            xn = self.xn
            yn = self.yn
            zn = self.zn
            magnetization = self.magnetization
        else:
            # According to https://docs.taichi-lang.org/docs/ndarray, or
            # https://github.com/taichi-dev/taichi/blob/8fdf7a7d/docs/lang/articles/basic/ndarray.md?plain=1#L23C35-L23C35
            # ti.ndarray-s are initialized to 0 by default.
            ret = ti_ndarray(dtype=ti.f64, shape=(n_rows, n_cols))
            xn = ti_ndarray_from(self.xn)
            yn = ti_ndarray_from(self.yn)
            zn = ti_ndarray_from(self.zn)
            magnetization = ti_ndarray_from(self.magnetization)
        magnetization_dim = self.magnetization.ndim

        # TODO: 由于Taichi目前采用i32作为索引类型，数组元素总数不能超过int32的上限，否则行为未定义
        #  如果后续Taichi支持i64索引类型，则可以去掉这个限制
        ratio = n_rows * n_cols / ti_size_max
        assert ratio < 1, (
            f"Kernel matrix with more than `INT32_MAX` elements has exceeded limits of Taichi"
            f" ({ratio:.2f}x larger than `INT32_MAX`)."
            f"\nConsider set `store_sensitivities='forward_only'`"
            f" or `builder.store_sensitivities(False)` if using `SimulationBuilder`."
            f"\nSee also `https://github.com/taichi-dev/taichi/issues/8161`."
        )

        monitor = None
        if progress is not None:
            progress.reset(total=n_rows * n_cols)

            # 用于配合正演的分批计算
            check_start_row = 0  # 该批正演的起点行号
            ending_row = 0  # 该批正演的终点行号
            processed = 0  # 已经处理的元素个数

            # 假设1: 计算出的值在绝大多数情况下不为0
            # 假设2: 因为正演算子是按行优先顺序计算的，绝大多数情况下最后一列的计算会在之前所有元素同时或之后完成的
            # 因此可以通过监视线程读取返回值矩阵ret的最后一列来估算进度，将单次估测复杂度从O(RC)降低到O(R)
            # 另: 若正演算子按列优先，则可以监视最后一行，保证监控变量的内存邻近，但是这会导致正演计算效率下降，因此未采用
            monitor = ValueObserver(lambda: processed + np.count_nonzero(ret[check_start_row:ending_row, -1]) * n_cols,
                                    lambda dx: progress.update(dx),
                                    interval=0.5)

            if is_cpu:
                # GPU版本暂无法通过该方法实现，因此在下面通过sync手动每批更新一次
                # CPU版本则启动监视线程
                monitor.start()

        start_row = 0
        for receiver in self.receivers:
            receiver_locations = receiver.receiver_locations
            components = receiver.components

            # TODO: 由于Taichi目前采用i32作为索引类型，
            #  主循环使用的ti.ndrange也会受限于这个限制，
            #  因此需要分批处理，如果后续Taichi支持i64索引类型，则可以去掉这个限制
            # 注：该分块依据是ti.ndrange循环的下标，且只对forward_only有效，核矩阵的大小仍然会收到这个限制
            # TODO: 对于GPU版，可能需要基于GPU显存大小进行批次划分
            n_rx_rows = receiver.n_rows
            n_batches = int(np.ceil((n_rx_rows * self.n_cells) / ti_size_max))

            all_components = np.asarray(['tmi', 'bx', 'by', 'bz', 'bxx', 'bxy', 'bxz', 'byy', 'byz', 'bzz'])
            components_indices = {f'i{c}': -1 for c in all_components}
            for i, c in enumerate(receiver.components):
                components_indices[f'i{c}'] = i
            n_components = len(components)

            for rx_locs in np.array_split(receiver_locations, n_batches):
                n_receivers = rx_locs.shape[0]
                section_rows = n_receivers * n_components

                if monitor is not None:
                    # 计算进度判定行区间
                    # 区间的更新和已处理元素的更新需要同时进行，防止observer给出错误结果
                    ending_row += section_rows
                    check_start_row = ending_row - section_rows
                    processed = start_row * n_cols

                self._kernel_matrix_forward(
                    xn=xn, yn=yn, zn=zn,
                    receiver_locations=rx_locs,
                    model=model, forward_only=forward_only,
                    magnetization=magnetization,
                    magnetization_dim=magnetization_dim,
                    **components_indices, n_components=n_components,
                    start_row=start_row,
                    ret=ret)

                start_row += section_rows

                if monitor is not None:
                    # 同步正确进度，防止observer读取不及时，统一进度
                    # 并且在GPU版中用于更新进度
                    monitor.sync(start_row * n_cols)

        if monitor is not None:
            monitor.stop()
            progress.close()

        if not is_cpu:
            ret = ret.to_numpy()

        return ret

    @ensure_contiguous
    @ti_kernel
    def _kernel_matrix_forward(
            self,
            xn: ti.types.ndarray(), yn: ti.types.ndarray(), zn: ti.types.ndarray(),
            receiver_locations: ti.types.ndarray(),
            itmi: ti.template(), ibx: ti.template(), iby: ti.template(), ibz: ti.template(),
            ibxx: ti.template(), ibxy: ti.template(), ibxz: ti.template(),
            ibyy: ti.template(), ibyz: ti.template(),
            ibzz: ti.template(),
            n_components: ti.template(),
            magnetization: ti.types.ndarray(),
            magnetization_dim: ti.template(),
            model: ti.types.ndarray(),
            forward_only: ti.template(),
            start_row: ti.i64,
            ret: ti.types.ndarray(),
    ):
        """计算正演结果或核矩阵

        Parameters
        ----------
        xn, yn, zn
            网格x,y,z方向上下界坐标
        receiver_locations
            观测点网格
        itmi, ibx, iby, ibz, ibxx, ibxy, ibxz, ibyy, ibyz, ibzz
            每个分量在单个观测点数据中的索引，-1代表不需要计算该分量
        n_components
            需要计算的分量数
        magnetization
            磁化矩阵，array(n_cells, 3) or array(3,) or array()
        magnetization_dim
            磁化矩阵维度
        model
            模型参数，scalar模型下为array(n_cells,)，vector模型下为array(n_cells, 3)
        forward_only
            是否只计算正演结果，若为真则返回正演结果，否则返回核矩阵
        start_row
            起始行，指定输出矩阵在ret中的起点行

        Returns
        -------
        ret
            用于接收返回值，可以是numpy、torch或paddle的数组也可以是taichi的数组

        Notes
        -----
            非forward_only模式下，model参数不会参与计算
        """

        tol1 = 1e-10  # Tolerance 1 for numerical stability over nodes and edges
        tol2 = 1e-4  # Tolerance 2 for numerical stability over nodes and edges

        # number of observations
        n_obs = receiver_locations.shape[0]

        # base cell dimensions
        min_hx = self.base_cell_sizes[0]
        min_hy = self.base_cell_sizes[1]
        min_hz = self.base_cell_sizes[2]

        # projection for tmi, tmi = [bx, by, bz] * tmi_projection
        px = self.tmi_projection[0]
        py = self.tmi_projection[1]
        pz = self.tmi_projection[2]

        for iobs, icell in ti.ndrange(n_obs, self.n_cells):
            # comp. pos. differences for tne, bsw nodes. Adjust if location within
            # tolerance of a node or edge
            dz2 = zn[icell, 1] - receiver_locations[iobs, 2]
            if ti.abs(dz2) / min_hz < tol2:
                dz2 = tol2 * min_hz
            dz1 = zn[icell, 0] - receiver_locations[iobs, 2]
            if ti.abs(dz1) / min_hz < tol2:
                dz1 = tol2 * min_hz

            dy2 = yn[icell, 1] - receiver_locations[iobs, 1]
            if ti.abs(dy2) / min_hy < tol2:
                dy2 = tol2 * min_hy
            dy1 = yn[icell, 0] - receiver_locations[iobs, 1]
            if ti.abs(dy1) / min_hy < tol2:
                dy1 = tol2 * min_hy

            dx2 = xn[icell, 1] - receiver_locations[iobs, 0]
            if ti.abs(dx2) / min_hx < tol2:
                dx2 = tol2 * min_hx
            dx1 = xn[icell, 0] - receiver_locations[iobs, 0]
            if ti.abs(dx1) / min_hx < tol2:
                dx1 = tol2 * min_hx

            # comp. squared diff
            dx2dx2 = dx2 ** 2.0
            dx1dx1 = dx1 ** 2.0

            dy2dy2 = dy2 ** 2.0
            dy1dy1 = dy1 ** 2.0

            dz2dz2 = dz2 ** 2.0
            dz1dz1 = dz1 ** 2.0

            # 2D radius component squared of corner nodes
            R1 = dy2dy2 + dx2dx2
            R2 = dy2dy2 + dx1dx1
            R3 = dy1dy1 + dx2dx2
            R4 = dy1dy1 + dx1dx1

            # radius to each cell node
            r1 = ti.sqrt(dz2dz2 + R2)
            r2 = ti.sqrt(dz2dz2 + R1)
            r3 = ti.sqrt(dz1dz1 + R1)
            r4 = ti.sqrt(dz1dz1 + R2)
            r5 = ti.sqrt(dz2dz2 + R3)
            r6 = ti.sqrt(dz2dz2 + R4)
            r7 = ti.sqrt(dz1dz1 + R4)
            r8 = ti.sqrt(dz1dz1 + R3)

            # compactify argument calculations
            arg1_ = dx1 + dy2 + r1
            arg1 = dy2 + dz2 + r1
            arg2 = dx1 + dz2 + r1
            arg3 = dx1 + r1
            arg4 = dy2 + r1
            arg5 = dz2 + r1

            arg6_ = dx2 + dy2 + r2
            arg6 = dy2 + dz2 + r2
            arg7 = dx2 + dz2 + r2
            arg8 = dx2 + r2
            arg9 = dy2 + r2
            arg10 = dz2 + r2

            arg11_ = dx2 + dy2 + r3
            arg11 = dy2 + dz1 + r3
            arg12 = dx2 + dz1 + r3
            arg13 = dx2 + r3
            arg14 = dy2 + r3
            arg15 = dz1 + r3

            arg16_ = dx1 + dy2 + r4
            arg16 = dy2 + dz1 + r4
            arg17 = dx1 + dz1 + r4
            arg18 = dx1 + r4
            arg19 = dy2 + r4
            arg20 = dz1 + r4

            arg21_ = dx2 + dy1 + r5
            arg21 = dy1 + dz2 + r5
            arg22 = dx2 + dz2 + r5
            arg23 = dx2 + r5
            arg24 = dy1 + r5
            arg25 = dz2 + r5

            arg26_ = dx1 + dy1 + r6
            arg26 = dy1 + dz2 + r6
            arg27 = dx1 + dz2 + r6
            arg28 = dx1 + r6
            arg29 = dy1 + r6
            arg30 = dz2 + r6

            arg31_ = dx1 + dy1 + r7
            arg31 = dy1 + dz1 + r7
            arg32 = dx1 + dz1 + r7
            arg33 = dx1 + r7
            arg34 = dy1 + r7
            arg35 = dz1 + r7

            arg36_ = dx2 + dy1 + r8
            arg36 = dy1 + dz1 + r8
            arg37 = dx2 + dz1 + r8
            arg38 = dx2 + r8
            arg39 = dy1 + r8
            arg40 = dz1 + r8

            if ti.static(ibx >= 0) or ti.static(itmi >= 0):
                tx = (
                    -2 * ti.atan2(dx1, arg1 + tol1)
                    - -2 * ti.atan2(dx2, arg6 + tol1)
                    + -2 * ti.atan2(dx2, arg11 + tol1)
                    - -2 * ti.atan2(dx1, arg16 + tol1)
                    + -2 * ti.atan2(dx2, arg21 + tol1)
                    - -2 * ti.atan2(dx1, arg26 + tol1)
                    + -2 * ti.atan2(dx1, arg31 + tol1)
                    - -2 * ti.atan2(dx2, arg36 + tol1)
                ) / -4 / ti.math.pi

                ty = (
                    ti.log(arg5)
                    - ti.log(arg10)
                    + ti.log(arg15)
                    - ti.log(arg20)
                    + ti.log(arg25)
                    - ti.log(arg30)
                    + ti.log(arg35)
                    - ti.log(arg40)
                ) / -4 / ti.math.pi

                tz = (
                    ti.log(arg4) - ti.log(arg9)
                    + ti.log(arg14) - ti.log(arg19)
                    + ti.log(arg24) - ti.log(arg29)
                    + ti.log(arg34) - ti.log(arg39)
                ) / -4 / ti.math.pi

                if ti.static(ibx >= 0):
                    self.write_to_ret(
                        tx=tx, ty=ty, tz=tz, ret=ret, start_row=start_row,
                        icell=icell, iobs=iobs, n_obs=n_obs, icomponent=ibx,
                        n_components=n_components,
                        model=model, forward_only=forward_only,
                        magnetization=magnetization, magnetization_dim=magnetization_dim,
                        accumulative=self.NonAccumulative
                    )

                if ti.static(itmi >= 0):
                    self.write_to_ret(
                        tx=px * tx, ty=px * ty, tz=px * tz, ret=ret, start_row=start_row,
                        icell=icell, iobs=iobs, n_obs=n_obs, icomponent=itmi,
                        n_components=n_components,
                        model=model, forward_only=forward_only,
                        magnetization=magnetization, magnetization_dim=magnetization_dim,
                        accumulative=self.Accumulative
                    )

            if ti.static(iby >= 0) or ti.static(itmi >= 0):
                tx = (
                    ti.log(arg5)
                    - ti.log(arg10)
                    + ti.log(arg15)
                    - ti.log(arg20)
                    + ti.log(arg25)
                    - ti.log(arg30)
                    + ti.log(arg35)
                    - ti.log(arg40)
                ) / -4 / ti.math.pi

                ty = (
                    -2 * ti.atan2(dy2, arg2 + tol1)
                    - -2 * ti.atan2(dy2, arg7 + tol1)
                    + -2 * ti.atan2(dy2, arg12 + tol1)
                    - -2 * ti.atan2(dy2, arg17 + tol1)
                    + -2 * ti.atan2(dy1, arg22 + tol1)
                    - -2 * ti.atan2(dy1, arg27 + tol1)
                    + -2 * ti.atan2(dy1, arg32 + tol1)
                    - -2 * ti.atan2(dy1, arg37 + tol1)
                ) / -4 / ti.math.pi

                tz = (
                    ti.log(arg3) - ti.log(arg8)
                    + ti.log(arg13) - ti.log(arg18)
                    + ti.log(arg23) - ti.log(arg28)
                    + ti.log(arg33) - ti.log(arg38)
                ) / -4 / ti.math.pi

                if ti.static(iby >= 0):
                    self.write_to_ret(
                        tx=tx, ty=ty, tz=tz, ret=ret, start_row=start_row,
                        icell=icell, iobs=iobs, n_obs=n_obs, icomponent=iby,
                        n_components=n_components,
                        model=model, forward_only=forward_only,
                        magnetization=magnetization, magnetization_dim=magnetization_dim,
                        accumulative=self.NonAccumulative
                    )

                if ti.static(itmi >= 0):
                    self.write_to_ret(
                        tx=py * tx, ty=py * ty, tz=py * tz, ret=ret, start_row=start_row,
                        icell=icell, iobs=iobs, n_obs=n_obs, icomponent=itmi,
                        n_components=n_components,
                        model=model, forward_only=forward_only,
                        magnetization=magnetization, magnetization_dim=magnetization_dim,
                        accumulative=self.Accumulative
                    )

            if ti.static(ibz >= 0) or ti.static(itmi >= 0):
                tx = (
                    ti.log(arg4)
                    - ti.log(arg9)
                    + ti.log(arg14)
                    - ti.log(arg19)
                    + ti.log(arg24)
                    - ti.log(arg29)
                    + ti.log(arg34)
                    - ti.log(arg39)
                ) / -4 / ti.math.pi

                ty = (
                    ti.log(arg3) - ti.log(arg8)
                    + ti.log(arg13) - ti.log(arg18)
                    + ti.log(arg23) - ti.log(arg28)
                    + ti.log(arg33) - ti.log(arg38)
                ) / -4 / ti.math.pi

                tz = (
                    -2 * ti.atan2(dz2, arg1_ + tol1)
                    - -2 * ti.atan2(dz2, arg6_ + tol1)
                    + -2 * ti.atan2(dz1, arg11_ + tol1)
                    - -2 * ti.atan2(dz1, arg16_ + tol1)
                    + -2 * ti.atan2(dz2, arg21_ + tol1)
                    - -2 * ti.atan2(dz2, arg26_ + tol1)
                    + -2 * ti.atan2(dz1, arg31_ + tol1)
                    - -2 * ti.atan2(dz1, arg36_ + tol1)
                ) / -4 / ti.math.pi

                if ti.static(ibz >= 0):
                    self.write_to_ret(
                        tx=tx, ty=ty, tz=tz, ret=ret, start_row=start_row,
                        icell=icell, iobs=iobs, n_obs=n_obs, icomponent=ibz,
                        n_components=n_components,
                        model=model, forward_only=forward_only,
                        magnetization=magnetization, magnetization_dim=magnetization_dim,
                        accumulative=self.NonAccumulative
                    )

                if ti.static(itmi >= 0):
                    self.write_to_ret(
                        tx=pz * tx, ty=pz * ty, tz=pz * tz, ret=ret, start_row=start_row,
                        icell=icell, iobs=iobs, n_obs=n_obs, icomponent=itmi,
                        n_components=n_components,
                        model=model, forward_only=forward_only,
                        magnetization=magnetization, magnetization_dim=magnetization_dim,
                        accumulative=self.Accumulative
                    )

            if ti.static(ibxx >= 0) or ti.static(ibzz >= 0):
                tx = 2 * (
                    ((dx1 ** 2 - r1 * arg1) / (r1 * arg1 ** 2 + dx1 ** 2 * r1))
                    - ((dx2 ** 2 - r2 * arg6) / (r2 * arg6 ** 2 + dx2 ** 2 * r2))
                    + ((dx2 ** 2 - r3 * arg11) / (r3 * arg11 ** 2 + dx2 ** 2 * r3))
                    - ((dx1 ** 2 - r4 * arg16) / (r4 * arg16 ** 2 + dx1 ** 2 * r4))
                    + ((dx2 ** 2 - r5 * arg21) / (r5 * arg21 ** 2 + dx2 ** 2 * r5))
                    - ((dx1 ** 2 - r6 * arg26) / (r6 * arg26 ** 2 + dx1 ** 2 * r6))
                    + ((dx1 ** 2 - r7 * arg31) / (r7 * arg31 ** 2 + dx1 ** 2 * r7))
                    - ((dx2 ** 2 - r8 * arg36) / (r8 * arg36 ** 2 + dx2 ** 2 * r8))
                ) / 4 / ti.math.pi

                ty = (
                    dx2 / (r5 * arg25)
                    - dx2 / (r2 * arg10)
                    + dx2 / (r3 * arg15)
                    - dx2 / (r8 * arg40)
                    + dx1 / (r1 * arg5)
                    - dx1 / (r6 * arg30)
                    + dx1 / (r7 * arg35)
                    - dx1 / (r4 * arg20)
                ) / 4 / ti.math.pi

                tz = (
                    dx1 / (r1 * arg4)
                    - dx2 / (r2 * arg9)
                    + dx2 / (r3 * arg14)
                    - dx1 / (r4 * arg19)
                    + dx2 / (r5 * arg24)
                    - dx1 / (r6 * arg29)
                    + dx1 / (r7 * arg34)
                    - dx2 / (r8 * arg39)
                ) / 4 / ti.math.pi

                if ti.static(ibxx >= 0):
                    self.write_to_ret(
                        tx=tx, ty=ty, tz=tz, ret=ret, start_row=start_row,
                        icell=icell, iobs=iobs, n_obs=n_obs, icomponent=ibxx,
                        n_components=n_components,
                        model=model, forward_only=forward_only,
                        magnetization=magnetization, magnetization_dim=magnetization_dim,
                        accumulative=self.NonAccumulative
                    )

                if ti.static(ibzz >= 0):
                    self.write_to_ret(
                        tx=-tx, ty=-ty, tz=-tz, ret=ret, start_row=start_row,
                        icell=icell, iobs=iobs, n_obs=n_obs, icomponent=ibzz,
                        n_components=n_components,
                        model=model, forward_only=forward_only,
                        magnetization=magnetization, magnetization_dim=magnetization_dim,
                        accumulative=self.Accumulative
                    )

            if ti.static(ibyy >= 0) or ti.static(ibzz >= 0):
                tx = (
                    dy2 / (r3 * arg15)
                    - dy2 / (r2 * arg10)
                    + dy1 / (r5 * arg25)
                    - dy1 / (r8 * arg40)
                    + dy2 / (r1 * arg5)
                    - dy2 / (r4 * arg20)
                    + dy1 / (r7 * arg35)
                    - dy1 / (r6 * arg30)
                ) / 4 / ti.math.pi

                ty = 2 * (
                    ((dy2 ** 2 - r1 * arg2) / (r1 * arg2 ** 2 + dy2 ** 2 * r1))
                    - ((dy2 ** 2 - r2 * arg7) / (r2 * arg7 ** 2 + dy2 ** 2 * r2))
                    + ((dy2 ** 2 - r3 * arg12) / (r3 * arg12 ** 2 + dy2 ** 2 * r3))
                    - ((dy2 ** 2 - r4 * arg17) / (r4 * arg17 ** 2 + dy2 ** 2 * r4))
                    + ((dy1 ** 2 - r5 * arg22) / (r5 * arg22 ** 2 + dy1 ** 2 * r5))
                    - ((dy1 ** 2 - r6 * arg27) / (r6 * arg27 ** 2 + dy1 ** 2 * r6))
                    + ((dy1 ** 2 - r7 * arg32) / (r7 * arg32 ** 2 + dy1 ** 2 * r7))
                    - ((dy1 ** 2 - r8 * arg37) / (r8 * arg37 ** 2 + dy1 ** 2 * r8))
                ) / 4 / ti.math.pi

                tz = (
                    dy2 / (r1 * arg3)
                    - dy2 / (r2 * arg8)
                    + dy2 / (r3 * arg13)
                    - dy2 / (r4 * arg18)
                    + dy1 / (r5 * arg23)
                    - dy1 / (r6 * arg28)
                    + dy1 / (r7 * arg33)
                    - dy1 / (r8 * arg38)
                ) / 4 / ti.math.pi

                if ti.static(ibyy >= 0):
                    self.write_to_ret(
                        tx=tx, ty=ty, tz=tz, ret=ret, start_row=start_row,
                        icell=icell, iobs=iobs, n_obs=n_obs, icomponent=ibyy,
                        n_components=n_components,
                        model=model, forward_only=forward_only,
                        magnetization=magnetization, magnetization_dim=magnetization_dim,
                        accumulative=self.NonAccumulative
                    )

                if ti.static(ibzz >= 0):
                    self.write_to_ret(
                        tx=-tx, ty=-ty, tz=-tz, ret=ret, start_row=start_row,
                        icell=icell, iobs=iobs, n_obs=n_obs, icomponent=ibzz,
                        n_components=n_components,
                        model=model, forward_only=forward_only,
                        magnetization=magnetization, magnetization_dim=magnetization_dim,
                        accumulative=self.Accumulative
                    )

            if ti.static(ibxy >= 0):
                tx = 2 * (
                    ((dx1 * arg4) / (r1 * arg1 ** 2 + (dx1 ** 2) * r1))
                    - ((dx2 * arg9) / (r2 * arg6 ** 2 + (dx2 ** 2) * r2))
                    + ((dx2 * arg14) / (r3 * arg11 ** 2 + (dx2 ** 2) * r3))
                    - ((dx1 * arg19) / (r4 * arg16 ** 2 + (dx1 ** 2) * r4))
                    + ((dx2 * arg24) / (r5 * arg21 ** 2 + (dx2 ** 2) * r5))
                    - ((dx1 * arg29) / (r6 * arg26 ** 2 + (dx1 ** 2) * r6))
                    + ((dx1 * arg34) / (r7 * arg31 ** 2 + (dx1 ** 2) * r7))
                    - ((dx2 * arg39) / (r8 * arg36 ** 2 + (dx2 ** 2) * r8))
                ) / 4 / ti.math.pi

                ty = (
                    dy2 / (r1 * arg5)
                    - dy2 / (r2 * arg10)
                    + dy2 / (r3 * arg15)
                    - dy2 / (r4 * arg20)
                    + dy1 / (r5 * arg25)
                    - dy1 / (r6 * arg30)
                    + dy1 / (r7 * arg35)
                    - dy1 / (r8 * arg40)
                ) / 4 / ti.math.pi

                tz = (
                    1 / r1 - 1 / r2 + 1 / r3 - 1 / r4 + 1 / r5 - 1 / r6 + 1 / r7 - 1 / r8
                ) / 4 / ti.math.pi

                self.write_to_ret(
                    tx=tx, ty=ty, tz=tz, ret=ret, start_row=start_row,
                    icell=icell, iobs=iobs, n_obs=n_obs, icomponent=ibxy,
                    n_components=n_components,
                    model=model, forward_only=forward_only,
                    magnetization=magnetization, magnetization_dim=magnetization_dim,
                    accumulative=self.NonAccumulative
                )

            if ti.static(ibxz >= 0):
                tx = 2 * (
                    ((dx1 * arg5) / (r1 * (arg1 ** 2) + (dx1 ** 2) * r1))
                    - ((dx2 * arg10) / (r2 * (arg6 ** 2) + (dx2 ** 2) * r2))
                    + ((dx2 * arg15) / (r3 * (arg11 ** 2) + (dx2 ** 2) * r3))
                    - ((dx1 * arg20) / (r4 * (arg16 ** 2) + (dx1 ** 2) * r4))
                    + ((dx2 * arg25) / (r5 * (arg21 ** 2) + (dx2 ** 2) * r5))
                    - ((dx1 * arg30) / (r6 * (arg26 ** 2) + (dx1 ** 2) * r6))
                    + ((dx1 * arg35) / (r7 * (arg31 ** 2) + (dx1 ** 2) * r7))
                    - ((dx2 * arg40) / (r8 * (arg36 ** 2) + (dx2 ** 2) * r8))
                ) / 4 / ti.math.pi

                ty = (
                    1 / r1 - 1 / r2 + 1 / r3 - 1 / r4 + 1 / r5 - 1 / r6 + 1 / r7 - 1 / r8
                ) / 4 / ti.math.pi

                tz = (
                    dz2 / (r1 * arg4)
                    - dz2 / (r2 * arg9)
                    + dz1 / (r3 * arg14)
                    - dz1 / (r4 * arg19)
                    + dz2 / (r5 * arg24)
                    - dz2 / (r6 * arg29)
                    + dz1 / (r7 * arg34)
                    - dz1 / (r8 * arg39)
                ) / 4 / ti.math.pi

                self.write_to_ret(
                    tx=tx, ty=ty, tz=tz, ret=ret, start_row=start_row,
                    icell=icell, iobs=iobs, n_obs=n_obs, icomponent=ibxz,
                    n_components=n_components,
                    model=model, forward_only=forward_only,
                    magnetization=magnetization, magnetization_dim=magnetization_dim,
                    accumulative=self.NonAccumulative
                )

            if ti.static(ibyz >= 0):
                tx = (
                    1 / r3 - 1 / r2 + 1 / r5 - 1 / r8 + 1 / r1 - 1 / r4 + 1 / r7 - 1 / r6
                ) / 4 / ti.math.pi

                ty = 2 * (
                    ((dy2 * arg5) / (r1 * (arg2 ** 2) + (dy2 ** 2) * r1))
                    - ((dy2 * arg10) / (r2 * (arg7 ** 2) + (dy2 ** 2) * r2))
                    + ((dy2 * arg15) / (r3 * (arg12 ** 2) + (dy2 ** 2) * r3))
                    - ((dy2 * arg20) / (r4 * (arg17 ** 2) + (dy2 ** 2) * r4))
                    + ((dy1 * arg25) / (r5 * (arg22 ** 2) + (dy1 ** 2) * r5))
                    - ((dy1 * arg30) / (r6 * (arg27 ** 2) + (dy1 ** 2) * r6))
                    + ((dy1 * arg35) / (r7 * (arg32 ** 2) + (dy1 ** 2) * r7))
                    - ((dy1 * arg40) / (r8 * (arg37 ** 2) + (dy1 ** 2) * r8))
                ) / 4 / ti.math.pi

                tz = (
                    dz2 / (r1 * arg3)
                    - dz2 / (r2 * arg8)
                    + dz1 / (r3 * arg13)
                    - dz1 / (r4 * arg18)
                    + dz2 / (r5 * arg23)
                    - dz2 / (r6 * arg28)
                    + dz1 / (r7 * arg33)
                    - dz1 / (r8 * arg38)
                ) / 4 / ti.math.pi

                self.write_to_ret(
                    tx=tx, ty=ty, tz=tz, ret=ret, start_row=start_row,
                    icell=icell, iobs=iobs, n_obs=n_obs, icomponent=ibyz,
                    n_components=n_components,
                    model=model, forward_only=forward_only,
                    magnetization=magnetization, magnetization_dim=magnetization_dim,
                    accumulative=self.NonAccumulative
                )

    @ti_func
    def write_to_ret(
            self,
            tx, ty, tz,
            ret: ti.types.ndarray(),
            start_row,
            icell,
            iobs, n_obs,
            icomponent: ti.template(),
            model: ti.types.ndarray(),
            magnetization: ti.types.ndarray(),
            magnetization_dim: ti.template(),
            n_components: ti.template(),
            forward_only: ti.template(),
            accumulative: ti.template(),
    ):
        """将正演结果写入输出矩阵

        Parameters
        ----------
        tx, ty, tz
            三方向系数
        ret
            输出矩阵
        start_row
            输出矩阵的起点行
        magnetization
            磁化矩阵，array(n_cells, 3) or array(3,) or array()
        magnetization_dim
            磁化矩阵维度
        model
            模型数组，如果为scalar模型则应为array(n_cells,)否则应为array(n_cells, 3)
        icell
            网格索引
        iobs
            观测点索引
        icomponent
            分量索引
        n_obs
            观测点数量
        n_components
            待求解分量数量
        forward_only
            是否只计算正演（直接乘以模型）
        accumulative
            是否累加（如果为False则会覆盖原有值），适用于forward_only或tmi的计算
        """
        i = self.in_col_index(iobs, icomponent, n_components, n_obs) + start_row

        # 根据磁化矩阵的维度选择计算方式
        if ti.static(magnetization_dim == 2):
            tx *= magnetization[icell, 0]
            ty *= magnetization[icell, 1]
            tz *= magnetization[icell, 2]
        elif ti.static(magnetization_dim == 1):
            tx *= magnetization[0]
            ty *= magnetization[1]
            tz *= magnetization[2]
        else:
            tx *= magnetization[None]
            ty *= magnetization[None]
            tz *= magnetization[None]

        if ti.static(self.model_type == 0):  # MType_Vector
            if ti.static(not forward_only):
                jx = self.in_row_index(icell, 0)
                jy = self.in_row_index(icell, 1)
                jz = self.in_row_index(icell, 2)
                if ti.static(accumulative == 0):  # NonAccumulative
                    ret[i, jx] = tx
                    ret[i, jy] = ty
                    ret[i, jz] = tz
                else:
                    ret[i, jx] += tx
                    ret[i, jy] += ty
                    ret[i, jz] += tz
            else:
                ret[i, 0] += model[icell, 0] * tx + model[icell, 1] * ty + model[icell, 2] * tz
        else:
            t = tx + ty + tz

            if ti.static(not forward_only):
                if ti.static(accumulative == 0):  # NonAccumulative
                    ret[i, icell] = t
                else:
                    ret[i, icell] += t
            else:
                ret[i, 0] += t * model[icell]

    @ti_func
    def in_col_index(self, iobs, icomponent, n_components, n_obs):
        return ti_index(iobs, icomponent, self.col_stype, n_components, n_obs)

    @ti_func
    def in_row_index(self, icell, bi):
        return ti_index(icell, bi, self.row_stype, 3, self.n_cells)
