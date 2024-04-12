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

        _init_table(*self.Tmat6, self.shape_cells, base_cell_sizes)

        kernel_matrix_forward(
            self.receiver_locations,
            self.xn, self.yn, self.zn,
            self.base_cell_sizes, model,
            *self.Tmat6, mat=np.empty(0), kernel_dtype=self.kernel_dt,
            write_to_mat=False, compressed=True,
            apply_susc_model=False
        )

    def solve(self, magnetization, model):
        if not self.use_complete_mesh and not np.all(self.active_cells_mask):
            indices_mask = np.full_like(self.active_cells_mask, -1, ti_size_dtype)
            indices_mask[self.active_cells_mask] = np.arange(np.count_nonzero(self.active_cells_mask))
        else:
            indices_mask = None

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
        base_cell_sizes: ti.types.ndarray()
):
    table_size = Txx.shape[0]
    nx, ny, nz = shape_cells[0], shape_cells[1], shape_cells[2]
    bx, by, bz = base_cell_sizes[0], base_cell_sizes[1], base_cell_sizes[2]

    for n in range(table_size):
        i, j, k = index_n2ijk(n, nx, ny, nz)

        Txx[n] = ti.cast(bx * (i - 0.5), Txx.dtype)
        Txy[n] = ti.cast(bx * (i + 0.5), Txy.dtype)
        Txz[n] = ti.cast(by * (j - 0.5), Txz.dtype)
        Tyy[n] = ti.cast(by * (j + 0.5), Tyy.dtype)
        Tyz[n] = ti.cast(bz * (k - 0.5), Tyz.dtype)
        Tzz[n] = ti.cast(bz * (k + 0.5), Tzz.dtype)


def solve_Tx_b_indexed(Tmat321, m, model, indices, shape_cells, progress: bool = False):
    index_dt = ti_size_t
    n_total_cells = np.prod(shape_cells)

    with ti_FieldsBuilder() as fields_builder:
        model, _model = ti_ndarray_like(model, field=True, builder=fields_builder), model

        if indices is not None:
            indices, _indices = ti_ndarray_like(indices, field=True, builder=fields_builder), indices
            use_active_cells_mapping = True
        else:
            use_active_cells_mapping = False

        fields_builder.finalize()

        if indices is not None:
            copy_from(indices, _indices)
        copy_from(model, _model)

        table_size = Tmat321[0][0].shape[0]
        nx, ny, nz = shape_cells

        @ti_pyfunc
        def transform(i):
            if ti.static(use_active_cells_mapping):
                # 映射到有效网格索引，如果返回-1则代表非有效网格
                return indices[i]
            else:
                return i

        @ti_pyfunc
        def extract(i, i_cell):
            neg_sus = -model[i_cell]
            return (
                neg_sus * Tmat321[0][0][i], neg_sus * Tmat321[0][1][i], neg_sus * Tmat321[0][2][i],
                neg_sus * Tmat321[1][0][i], neg_sus * Tmat321[1][1][i],
                neg_sus * Tmat321[2][0][i]
            )

        @ti_pyfunc
        def iterate_relation_indices(x: ti.template(), Ax: ti.template(), upper: ti.template()):
            for n in range(1, table_size):
                i, j, k = index_n2ijk(n, nx, ny, nz)
                delta = ti.cast(i + nx * j + nx * ny * k, index_dt)

                i0 = 0 if i >= 0 else -i
                j0 = 0 if j >= 0 else -j
                k0 = 0 if k >= 0 else -k

                r0 = ti.cast(i0 + nx * j0 + nx * ny * k0, index_dt)

                di = ti.cast(nx - abs(i), index_dt)
                dj = ti.cast(ny - abs(j), index_dt)
                dk = ti.cast(nz - abs(k), index_dt)

                si = ti.cast(1, index_dt)
                sj = ti.cast(nx - di, index_dt)
                sk = ti.cast(nx * ny - dj * nx, index_dt)

                for kk in range(dk):
                    for jj in range(dj):
                        for ii in range(di):
                            r = transform(r0)
                            c = transform(r0 + delta)

                            if r >= 0 and c >= 0:
                                if ti.static(upper):
                                    r, c = c, r

                                row, col = r * 3, c * 3

                                txx, txy, txz, tyy, tyz, tzz = extract(n, c)
                                mx, my, mz = x[col + 0], x[col + 1], x[col + 2]
                                Ax[row + 0] += txx * mx + txy * my + txz * mz
                                Ax[row + 1] += txy * mx + tyy * my + tyz * mz
                                Ax[row + 2] += txz * mx + tyz * my + tzz * mz

                            r0 += si
                        r0 += sj
                    r0 += sk

        @ti_kernel
        def linear(x: ti.template(), Ax: ti.template()):
            for _cr in range(n_total_cells):
                cr = transform(_cr)
                if cr < 0:
                    continue

                txx, txy, txz, tyy, tyz, tzz = extract(0, cr)
                txx += 1
                tyy += 1
                tzz += 1

                cr = cr * 3
                mx, my, mz = x[cr + 0], x[cr + 1], x[cr + 2]
                Ax[cr + 0] = txx * mx + txy * my + txz * mz
                Ax[cr + 1] = txy * mx + tyy * my + tyz * mz
                Ax[cr + 2] = txz * mx + tyz * my + tzz * mz

            iterate_relation_indices(x, Ax, upper=True)
            iterate_relation_indices(x, Ax, upper=False)

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
