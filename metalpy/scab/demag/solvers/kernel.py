import taichi as ti

from metalpy.utils.taichi import ti_kernel
from metalpy.utils.taichi_kernels import ti_use


@ti_kernel
def kernel_matrix_forward(
        receiver_locations: ti.types.ndarray(),
        xn: ti.types.ndarray(),
        yn: ti.types.ndarray(),
        zn: ti.types.ndarray(),
        susc_model: ti.types.ndarray(),
        Txx: ti.template(),
        Txy: ti.template(),
        Txz: ti.template(),
        Tyy: ti.template(),
        Tyz: ti.template(),
        Tzz: ti.template(),
        mat: ti.types.ndarray(),
        write_to_mat: ti.template(),
        compressed: ti.template(),
        kernel_dtype: ti.template(),
        apply_susc_model: ti.template(),
):
    # calculates A = I - X @ T, where T is the forward kernel, s.t. T @ m_v = B_v
    # m_v and B_v are both channel first (Array of Structure in taichi)
    # m_v = [Mx1, My1, Mz1, ... Mxn, Myn, Mzn]
    # B_v = [Bx1, By1, Bz1, ... Bxn, Byn, Bzn]
    #     ┌─────────────────────────── nC ─────────────────────────┐
    #     ┌────────────────┬────────────────┬─────┬────────────────┐ ─┐
    #     │ Txx, Txy, Txz, │ Txx, Txy, Txz, │ ... │ Txx, Txy, Txz, │  │
    #     │ Tyx, Tyy, Tyz, │ Tyx, Tyy, Tyz, │ ... │ Tyx, Tyy, Tyz, │  │
    #     │ Tzx, Tzy, Tzz, │ Tzx, Tzy, Tzz, │ ... │ Tzx, Tzy, Tzz, │  │
    #     ├────────────────┼────────────────┼─────┼────────────────┤  │
    # T = │ ...            │ ...            │ ... │ ...            │ nObs
    #     ├────────────────┼────────────────┼─────┼────────────────┤  │
    #     │ Txx, Txy, Txz, │ Txx, Txy, Txz, │ ... │ Txx, Txy, Txz, │  │
    #     │ Tyx, Tyy, Tyz, │ Tyx, Tyy, Tyz, │ ... │ Tyx, Tyy, Tyz, │  │
    #     │ Tzx, Tzy, Tzz, │ Tzx, Tzy, Tzz, │ ... │ Tzx, Tzy, Tzz, │  │
    #     └────────────────┴────────────────┴─────┴────────────────┘ ─┘
    # number of cells in mesh
    nC = xn.shape[0]
    nObs = receiver_locations.shape[0]

    dummy = xn[0, 0]
    total = nObs * nC
    if ti.static(compressed):
        total = Txx.shape[0]

    for i in range(total):
        iobs, icell = 0, 0
        if ti.static(not compressed):
            iobs, icell = i // nC, i % nC

        dx1, dx2, dy1, dy2, dz1, dz2 = dummy, dummy, dummy, dummy, dummy, dummy
        ti_use(dx1, dx2, dy1, dy2, dz1, dz2)

        if ti.static(not compressed):
            dx1 = xn[icell, 0] - receiver_locations[iobs, 0]
            dx2 = xn[icell, 1] - receiver_locations[iobs, 0]
            dy1 = yn[icell, 0] - receiver_locations[iobs, 1]
            dy2 = yn[icell, 1] - receiver_locations[iobs, 1]
            dz1 = zn[icell, 0] - receiver_locations[iobs, 2]
            dz2 = zn[icell, 1] - receiver_locations[iobs, 2]
        else:
            dx1 = Txx[i]
            dx2 = Txy[i]
            dy1 = Txz[i]
            dy2 = Tyy[i]
            dz1 = Tyz[i]
            dz2 = Tzz[i]

        txx = txy = txz = tyy = tyz = tzz = dx1 * 0.0
        inside: ti.i8 = 0

        if not ti.math.isinf(dx1):
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
            arg1 = dy2 + dz2 + r1
            arg2 = dx1 + dz2 + r1
            arg3 = dx1 + r1
            arg4 = dy2 + r1
            arg5 = dz2 + r1

            arg6 = dy2 + dz2 + r2
            arg7 = dx2 + dz2 + r2
            arg8 = dx2 + r2
            arg9 = dy2 + r2
            arg10 = dz2 + r2

            arg11 = dy2 + dz1 + r3
            arg12 = dx2 + dz1 + r3
            arg13 = dx2 + r3
            arg14 = dy2 + r3
            arg15 = dz1 + r3

            arg16 = dy2 + dz1 + r4
            arg17 = dx1 + dz1 + r4
            arg18 = dx1 + r4
            arg19 = dy2 + r4
            arg20 = dz1 + r4

            arg21 = dy1 + dz2 + r5
            arg22 = dx2 + dz2 + r5
            arg23 = dx2 + r5
            arg24 = dy1 + r5
            arg25 = dz2 + r5

            arg26 = dy1 + dz2 + r6
            arg27 = dx1 + dz2 + r6
            arg28 = dx1 + r6
            arg29 = dy1 + r6
            arg30 = dz2 + r6

            arg31 = dy1 + dz1 + r7
            arg32 = dx1 + dz1 + r7
            arg33 = dx1 + r7
            arg34 = dy1 + r7
            arg35 = dz1 + r7

            arg36 = dy1 + dz1 + r8
            arg37 = dx2 + dz1 + r8
            arg38 = dx2 + r8
            arg39 = dy1 + r8
            arg40 = dz1 + r8

            # 判断观测点是否在网格内，如果在网格内，则对角线元素+1
            inside = ti.cast(
                + ti.math.sign(dx1 * dx2)
                + ti.math.sign(dy1 * dy2)
                + ti.math.sign(dz1 * dz2)
                == -3, ti.i8
            )

            txx = (
                + -2 * ti.atan2(dx1, arg1)
                - -2 * ti.atan2(dx2, arg6)
                + -2 * ti.atan2(dx2, arg11)
                - -2 * ti.atan2(dx1, arg16)
                + -2 * ti.atan2(dx2, arg21)
                - -2 * ti.atan2(dx1, arg26)
                + -2 * ti.atan2(dx1, arg31)
                - -2 * ti.atan2(dx2, arg36)
            ) / -4 / ti.math.pi

            txy = (
                + ti.log(arg5) - ti.log(arg10)
                + ti.log(arg15) - ti.log(arg20)
                + ti.log(arg25) - ti.log(arg30)
                + ti.log(arg35) - ti.log(arg40)
            ) / -4 / ti.math.pi

            txz = (
                + ti.log(arg4) - ti.log(arg9)
                + ti.log(arg14) - ti.log(arg19)
                + ti.log(arg24) - ti.log(arg29)
                + ti.log(arg34) - ti.log(arg39)
            ) / -4 / ti.math.pi

            tyy = (
                + -2 * ti.atan2(dy2, arg2)
                - -2 * ti.atan2(dy2, arg7)
                + -2 * ti.atan2(dy2, arg12)
                - -2 * ti.atan2(dy2, arg17)
                + -2 * ti.atan2(dy1, arg22)
                - -2 * ti.atan2(dy1, arg27)
                + -2 * ti.atan2(dy1, arg32)
                - -2 * ti.atan2(dy1, arg37)
            ) / -4 / ti.math.pi

            tyz = (
                + ti.log(arg3) - ti.log(arg8)
                + ti.log(arg13) - ti.log(arg18)
                + ti.log(arg23) - ti.log(arg28)
                + ti.log(arg33) - ti.log(arg38)
            ) / -4 / ti.math.pi

            # txx + tyy + tzz = 0, 如果观测点在网格外
            # txx + tyy + tzz = -1, 如果观测点在网格内
            tzz = -inside - txx - tyy

        tyx = txy
        tzx = txz
        tzy = tyz

        neg_sus = -susc_model[icell]

        if ti.static(not apply_susc_model):
            # 例如对于压缩核矩阵的算法，在非均匀的情况下不能提前乘以磁化率
            # 而需要求解时现场计算实际的系数矩阵
            neg_sus = 1
            inside = ti.cast(0, ti.i8)

        if ti.static(compressed):
            Txx[i] = ti.cast(neg_sus * txx + inside, kernel_dtype)
            Txy[i] = ti.cast(neg_sus * txy, kernel_dtype)
            Txz[i] = ti.cast(neg_sus * txz, kernel_dtype)
            Tyy[i] = ti.cast(neg_sus * tyy + inside, kernel_dtype)
            Tyz[i] = ti.cast(neg_sus * tyz, kernel_dtype)
            Tzz[i] = ti.cast(neg_sus * tzz + inside, kernel_dtype)
        else:
            if ti.static(write_to_mat):
                mat[iobs * 3 + 0, icell * 3 + 0] = ti.cast(neg_sus * txx + inside, kernel_dtype)
                mat[iobs * 3 + 0, icell * 3 + 1] = ti.cast(neg_sus * txy, kernel_dtype)
                mat[iobs * 3 + 0, icell * 3 + 2] = ti.cast(neg_sus * txz, kernel_dtype)
                mat[iobs * 3 + 1, icell * 3 + 0] = ti.cast(neg_sus * tyx, kernel_dtype)
                mat[iobs * 3 + 1, icell * 3 + 1] = ti.cast(neg_sus * tyy + inside, kernel_dtype)
                mat[iobs * 3 + 1, icell * 3 + 2] = ti.cast(neg_sus * tyz, kernel_dtype)
                mat[iobs * 3 + 2, icell * 3 + 0] = ti.cast(neg_sus * tzx, kernel_dtype)
                mat[iobs * 3 + 2, icell * 3 + 1] = ti.cast(neg_sus * tzy, kernel_dtype)
                mat[iobs * 3 + 2, icell * 3 + 2] = ti.cast(neg_sus * tzz + inside, kernel_dtype)
            else:
                Txx[iobs, icell] = ti.cast(neg_sus * txx + inside, kernel_dtype)
                Txy[iobs, icell] = ti.cast(neg_sus * txy, kernel_dtype)
                Txz[iobs, icell] = ti.cast(neg_sus * txz, kernel_dtype)
                Tyy[iobs, icell] = ti.cast(neg_sus * tyy + inside, kernel_dtype)
                Tyz[iobs, icell] = ti.cast(neg_sus * tyz, kernel_dtype)
                Tzz[iobs, icell] = ti.cast(neg_sus * tzz + inside, kernel_dtype)
