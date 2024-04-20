import warnings

import numpy as np
import taichi as ti

from metalpy.utils.taichi import ti_field, ti_FieldsBuilder, ti_kernel, ti_pyfunc, ti_ndarray_like, copy_from, \
    ti_size_t, ti_size_dtype
from metalpy.utils.taichi_kernels import ti_use
from metalpy.utils.ti_solvers import matrix_free
from .demag_solver_context import DemagSolverContext
from .kernel import kernel_matrix_forward
from .solver import DemagnetizationSolver


class IndexedSolver(DemagnetizationSolver):
    def __init__(
            self,
            context: DemagSolverContext
    ):
        """该函数通过关系索引来实现核矩阵的一种稀疏表示，从而降低内存需求，并实现支持任意大小的规则网格

        Parameters
        ----------
        context
            退磁求解上下文

        Notes
        -----
        优势：

        - 可以大幅降低内存需求量至O(n)

        缺陷：

        - 只支持在规则网格上使用
        - 对计算效率有较大影响
        """
        super().__init__(context)

        if not context.is_symmetric:
            warnings.warn(
                f'`{IndexedSolver.__name__}` requires symmetric mesh (got non-symmetric mesh),'
                f' may result in incorrect results.'
            )

        table_size = index_mat_size(*self.shape_cells)

        self.builder = builder = ti_FieldsBuilder()

        self.Tmat321 = [
            [ti_field(self.kernel_dtype) for _ in range(3)],  # Txx, Txy, Txz
            [ti_field(self.kernel_dtype) for _ in range(2)],  # ---- Tyy, Tyz
            [ti_field(self.kernel_dtype) for _ in range(1)],  # --------- Tzz
        ]
        self.Tmat6 = [t for ts in self.Tmat321 for t in ts]

        builder.dense(ti.i, table_size).place(*self.Tmat6)

        builder.finalize()

    def build_kernel(self, model):
        base_cell_sizes = self.base_cell_sizes

        _init_table(
            *self.Tmat6,
            self.shape_cells,
            base_cell_sizes,
            self.cutoff
        )

        if np.allclose(model, model[0]):
            apply_susc_model = True
        else:
            apply_susc_model = False

        kernel_matrix_forward(
            self.receiver_locations,
            self.xn, self.yn, self.zn,
            self.base_cell_sizes, model,
            *self.Tmat6, mat=np.empty(0), kernel_dtype=self.kernel_dt,
            write_to_mat=False, compressed=True,
            apply_susc_model=apply_susc_model
        )

    def solve(self, magnetization, model):
        if not self.use_complete_mesh and not np.all(self.active_cells_mask):
            indices_mask = np.full_like(self.active_cells_mask, -1, ti_size_dtype)
            indices_mask[self.active_cells_mask] = np.arange(np.count_nonzero(self.active_cells_mask))
        else:
            indices_mask = None

        if np.allclose(model, model[0]):
            model = None

        return solve_Tx_b_indexed(
            self.Tmat321,
            magnetization,
            model,
            indices_mask,
            shape_cells=self.shape_cells,
            progress=self.progress
        )


@ti_kernel
def _init_table(
        Txx: ti.template(),
        Txy: ti.template(),
        Txz: ti.template(),
        Tyy: ti.template(),
        Tyz: ti.template(),
        Tzz: ti.template(),
        shape_cells: ti.types.ndarray(),
        base_cell_sizes: ti.types.ndarray(),
        cutoff: ti.template()
):
    table_size = Txx.shape[0]
    nx, ny, nz = shape_cells[0], shape_cells[1], shape_cells[2]
    bx, by, bz = base_cell_sizes[0], base_cell_sizes[1], base_cell_sizes[2]

    for n in range(table_size):
        i, j, k = index_n2ijk(n, nx, ny, nz)

        if ti.static(cutoff != ti.math.inf):
            dist = ti.sqrt(
                (bx * i) ** 2
                + (by * j) ** 2
                + (bz * k) ** 2
            )

            if dist > cutoff:
                Txx[n] = Txy[n] = Txz[n] = Tyy[n] = Tyz[n] = Tzz[n] = ti.math.inf
                continue

        Txx[n] = ti.cast(bx * (i - 0.5), Txx.dtype)
        Txy[n] = ti.cast(bx * (i + 0.5), Txy.dtype)
        Txz[n] = ti.cast(by * (j - 0.5), Txz.dtype)
        Tyy[n] = ti.cast(by * (j + 0.5), Tyy.dtype)
        Tyz[n] = ti.cast(bz * (k - 0.5), Tyz.dtype)
        Tzz[n] = ti.cast(bz * (k + 0.5), Tzz.dtype)


def solve_Tx_b_indexed(Tmat321, m, model, indices, shape_cells, progress: bool = False):
    n_total_cells = n_total_obs = np.prod(shape_cells)

    with ti_FieldsBuilder() as fields_builder:
        if model is not None:
            model, _model = ti_ndarray_like(model, field=True, builder=fields_builder), model
            use_model = True
        else:
            use_model = False

        if indices is not None:
            indices, _indices = ti_ndarray_like(indices, field=True, builder=fields_builder), indices
            use_active_cells_mapping = True
        else:
            use_active_cells_mapping = False

        fields_builder.finalize()

        if indices is not None:
            copy_from(indices, _indices)

        if model is not None:
            copy_from(model, _model)

        nx, ny, nz = shape_cells
        nxny = nx * ny

        @ti_pyfunc
        def transform(i):
            if ti.static(use_active_cells_mapping):
                # 映射到有效网格索引，如果返回-1则代表非有效网格
                return indices[i]
            else:
                return i

        @ti_pyfunc
        def index3d(i):
            # 完整索引矩阵中的每一行对应一个网格到所有其它网格的关系，该关系由 nz 层2维网格组成，每层由 ny 条1维网格组成
            # 即包含 nz 个长度为 nxny 的大递增块，块间距为 (2 * nx - 1) * (2 * ny - 1)
            # 每个大递增块中包含 ny 个长度为 nx 的小递增块，块间距为 (2 * nx - 1)
            # 例如 nx=2, ny=3, nz=4
            # 完整索引矩阵的第一行即为
            # [0,  1,  3,  4,  6,  7,
            #  15, 16, 18, 19, 21, 22,
            #  30, 31, 33, 34, 36, 37,
            #  45, 46, 48, 49, 51, 52]
            p2, m2 = i // nxny, i % nxny
            p1, m1 = m2 // nx, m2 % nx

            return p2 * (2 * nx - 1) * (2 * ny - 1) + p1 * (2 * nx - 1) + m1

        @ti_pyfunc
        def ij2n(i, j):
            return ti.abs(index3d(j) - index3d(i))

        @ti_pyfunc
        def extract(i, i_cell):
            neg_sus: Tmat321[0][0].dtype = 1
            if ti.static(use_model):
                neg_sus = -model[i_cell]

            return (
                neg_sus * Tmat321[0][0][i], neg_sus * Tmat321[0][1][i], neg_sus * Tmat321[0][2][i],
                neg_sus * Tmat321[1][0][i], neg_sus * Tmat321[1][1][i],
                neg_sus * Tmat321[2][0][i]
            )

        @ti_kernel
        def linear(x: ti.template(), Ax: ti.template()):
            for r_g in range(n_total_cells):
                r = transform(r_g)  # 转为有效网格坐标
                if r < 0:
                    continue

                txx, txy, txz, tyy, tyz, tzz = extract(0, r)  # 对角线元素对应0号关系
                if ti.static(use_model):
                    txx += 1
                    tyy += 1
                    tzz += 1

                row = r * 3
                mx, my, mz = x[row + 0], x[row + 1], x[row + 2]

                summed0 = txx * mx + txy * my + txz * mz
                summed1 = txy * mx + tyy * my + tyz * mz
                summed2 = txz * mx + tyz * my + tzz * mz

                for c_g in range(n_total_obs):
                    if c_g == r_g:
                        # 对角线元素已经计入
                        continue

                    c = transform(c_g)  # 转为有效网格坐标
                    if c < 0:
                        continue

                    n = ij2n(r_g, c_g)  # 通过全局坐标获取关系索引序号
                    txx, txy, txz, tyy, tyz, tzz = extract(n, c)

                    col = c * 3
                    mx, my, mz = x[col + 0], x[col + 1], x[col + 2]

                    summed0 += txx * mx + txy * my + txz * mz
                    summed1 += txy * mx + tyy * my + tyz * mz
                    summed2 += txz * mx + tyz * my + tzz * mz

                Ax[row + 0] = summed0
                Ax[row + 1] = summed1
                Ax[row + 2] = summed2

        return matrix_free.cg(ti.linalg.LinearOperator(linear), m, progress=progress).to_numpy()


@ti_pyfunc
def index_mat_size(nx, ny, nz):
    return 1 + nx - 1 + (2 * nx - 1) * (ny - 1) + (2 * nx - 1) * (2 * ny - 1) * (nz - 1)


@ti_pyfunc
def index_n2ijk(n, nx, ny, nz):
    lv0 = nx - 1
    lv1 = (nx * 2 - 1) * (ny - 1)
    lv1_2 = lv0 + lv1
    nnx = nx * 2 - 1
    nny = ny * 2 - 1

    i: ti.int64 = 0
    j: ti.int64 = 0
    k: ti.int64 = 0

    ti_use(i)
    ti_use(j)
    ti_use(k)

    if n > lv1_2:
        res = n - lv0 - lv1 - 1
        i = res % nnx - nx + 1
        j = res // nnx % nny - ny + 1
        k = res // nnx // nny + 1
    elif n > lv0:
        res = n - lv0 - 1
        i = res % nnx - nx + 1
        j = res // nnx + 1
        k = 0
    else:
        i = n
        j = 0
        k = 0

    return i, j, k


@ti_pyfunc
def index_ijk2n(i, j, k, nx, ny, nz):
    lv0 = nx - 1
    lv1 = (nx * 2 - 1) * (ny - 1)
    ri = i + nx
    rj = j + ny
    nnx = nx * 2 - 1
    nny = ny * 2 - 1
    if k > 0:
        return lv0 + lv1 + nnx * nny * (k - 1) + nnx * (rj - 1) + ri
    elif j > 0:
        return lv0 + nnx * (j - 1) + ri
    else:
        return i
