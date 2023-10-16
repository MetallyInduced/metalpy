import taichi as ti

from metalpy.utils.taichi import ti_kernel
from metalpy.utils.taichi_kernels import ti_use


@ti_kernel
def kernel_matrix_forward(
        receiver_locations: ti.types.ndarray(),
        xn: ti.types.ndarray(),
        yn: ti.types.ndarray(),
        zn: ti.types.ndarray(),
        base_cell_sizes: ti.types.ndarray(),
        susc_model: ti.types.ndarray(),
        Txx: ti.template(),
        Txy: ti.template(),
        Txz: ti.template(),
        Tyx: ti.template(),
        Tyy: ti.template(),
        Tyz: ti.template(),
        Tzx: ti.template(),
        Tzy: ti.template(),
        Tzz: ti.template(),
        mat: ti.types.ndarray(),
        write_to_mat: ti.template(),
        compressed: ti.template(),
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
    tol1 = 1e-10  # Tolerance 1 for numerical stability over nodes and edges
    tol2 = 1e-4  # Tolerance 2 for numerical stability over nodes and edges

    # number of cells in mesh
    nC = xn.shape[0]
    nObs = receiver_locations.shape[0]

    # base cell dimensions
    min_hx = base_cell_sizes[0]
    min_hy = base_cell_sizes[1]
    min_hz = base_cell_sizes[2]

    dummy = xn[0, 0]
    total = nObs * nC
    if ti.static(compressed):
        total = Txx.shape[0]

    for i in range(total):
        iobs, icell = 0, 0
        if ti.static(not compressed):
            iobs, icell = i // nC, i % nC

        # comp. pos. differences for tne, bsw nodes. Adjust if location within
        # tolerance of a node or edge
        dx1, dx2, dy1, dy2, dz1, dz2 = dummy, dummy, dummy, dummy, dummy, dummy
        ti_use(dx1)
        ti_use(dx2)
        ti_use(dy1)
        ti_use(dy2)
        ti_use(dz1)
        ti_use(dz2)

        if ti.static(not compressed):
            dx1 = xn[icell, 0] - receiver_locations[iobs, 0]
            dx2 = xn[icell, 1] - receiver_locations[iobs, 0]
            dy1 = yn[icell, 0] - receiver_locations[iobs, 1]
            dy2 = yn[icell, 1] - receiver_locations[iobs, 1]
            dz1 = zn[icell, 0] - receiver_locations[iobs, 2]
            dz2 = zn[icell, 1] - receiver_locations[iobs, 2]
        else:
            dx1 = Txx[i]

            if ti.math.isnan(dx1):
                continue

            dx2 = Txy[i]
            dy1 = Txz[i]
            dy2 = Tyx[i]
            dz1 = Tyy[i]
            dz2 = Tyz[i]

        if ti.abs(dx1) / min_hx < tol2:
            dx1 = tol2 * min_hx
        if ti.abs(dx2) / min_hx < tol2:
            dx2 = tol2 * min_hx

        if ti.abs(dy1) / min_hy < tol2:
            dy1 = tol2 * min_hy
        if ti.abs(dy2) / min_hy < tol2:
            dy2 = tol2 * min_hy

        if ti.abs(dz1) / min_hz < tol2:
            dz1 = tol2 * min_hz
        if ti.abs(dz2) / min_hz < tol2:
            dz2 = tol2 * min_hz

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

        txx = (
            -2 * ti.atan2(dx1, arg1 + tol1)
            - -2 * ti.atan2(dx2, arg6 + tol1)
            + -2 * ti.atan2(dx2, arg11 + tol1)
            - -2 * ti.atan2(dx1, arg16 + tol1)
            + -2 * ti.atan2(dx2, arg21 + tol1)
            - -2 * ti.atan2(dx1, arg26 + tol1)
            + -2 * ti.atan2(dx1, arg31 + tol1)
            - -2 * ti.atan2(dx2, arg36 + tol1)
        ) / -4 / ti.math.pi

        txy = (
            ti.log(arg5)
            - ti.log(arg10)
            + ti.log(arg15)
            - ti.log(arg20)
            + ti.log(arg25)
            - ti.log(arg30)
            + ti.log(arg35)
            - ti.log(arg40)
        ) / -4 / ti.math.pi

        txz = (
            ti.log(arg4) - ti.log(arg9)
            + ti.log(arg14) - ti.log(arg19)
            + ti.log(arg24) - ti.log(arg29)
            + ti.log(arg34) - ti.log(arg39)
        ) / -4 / ti.math.pi

        tyx = (
            ti.log(arg5)
            - ti.log(arg10)
            + ti.log(arg15)
            - ti.log(arg20)
            + ti.log(arg25)
            - ti.log(arg30)
            + ti.log(arg35)
            - ti.log(arg40)
        ) / -4 / ti.math.pi

        tyy = (
            -2 * ti.atan2(dy2, arg2 + tol1)
            - -2 * ti.atan2(dy2, arg7 + tol1)
            + -2 * ti.atan2(dy2, arg12 + tol1)
            - -2 * ti.atan2(dy2, arg17 + tol1)
            + -2 * ti.atan2(dy1, arg22 + tol1)
            - -2 * ti.atan2(dy1, arg27 + tol1)
            + -2 * ti.atan2(dy1, arg32 + tol1)
            - -2 * ti.atan2(dy1, arg37 + tol1)
        ) / -4 / ti.math.pi

        tyz = (
            ti.log(arg3) - ti.log(arg8)
            + ti.log(arg13) - ti.log(arg18)
            + ti.log(arg23) - ti.log(arg28)
            + ti.log(arg33) - ti.log(arg38)
        ) / -4 / ti.math.pi

        tzx = (
            ti.log(arg4)
            - ti.log(arg9)
            + ti.log(arg14)
            - ti.log(arg19)
            + ti.log(arg24)
            - ti.log(arg29)
            + ti.log(arg34)
            - ti.log(arg39)
        ) / -4 / ti.math.pi

        tzy = (
            ti.log(arg3) - ti.log(arg8)
            + ti.log(arg13) - ti.log(arg18)
            + ti.log(arg23) - ti.log(arg28)
            + ti.log(arg33) - ti.log(arg38)
        ) / -4 / ti.math.pi

        tzz = (
            -2 * ti.atan2(dz2, arg1_ + tol1)
            - -2 * ti.atan2(dz2, arg6_ + tol1)
            + -2 * ti.atan2(dz1, arg11_ + tol1)
            - -2 * ti.atan2(dz1, arg16_ + tol1)
            + -2 * ti.atan2(dz2, arg21_ + tol1)
            - -2 * ti.atan2(dz2, arg26_ + tol1)
            + -2 * ti.atan2(dz1, arg31_ + tol1)
            - -2 * ti.atan2(dz1, arg36_ + tol1)
        ) / -4 / ti.math.pi

        neg_sus = -susc_model[icell]

        if ti.static(compressed):
            Txx[i] = neg_sus * txx
            Txy[i] = neg_sus * txy
            Txz[i] = neg_sus * txz
            Tyx[i] = neg_sus * tyx
            Tyy[i] = neg_sus * tyy
            Tyz[i] = neg_sus * tyz
            Tzx[i] = neg_sus * tzx
            Tzy[i] = neg_sus * tzy
            Tzz[i] = neg_sus * tzz
        else:
            if ti.static(write_to_mat):
                mat[iobs * 3 + 0, icell * 3 + 0] = neg_sus * txx
                mat[iobs * 3 + 0, icell * 3 + 1] = neg_sus * txy
                mat[iobs * 3 + 0, icell * 3 + 2] = neg_sus * txz
                mat[iobs * 3 + 1, icell * 3 + 0] = neg_sus * tyx
                mat[iobs * 3 + 1, icell * 3 + 1] = neg_sus * tyy
                mat[iobs * 3 + 1, icell * 3 + 2] = neg_sus * tyz
                mat[iobs * 3 + 2, icell * 3 + 0] = neg_sus * tzx
                mat[iobs * 3 + 2, icell * 3 + 1] = neg_sus * tzy
                mat[iobs * 3 + 2, icell * 3 + 2] = neg_sus * tzz
            else:
                Txx[iobs, icell] = neg_sus * txx
                Txy[iobs, icell] = neg_sus * txy
                Txz[iobs, icell] = neg_sus * txz
                Tyx[iobs, icell] = neg_sus * tyx
                Tyy[iobs, icell] = neg_sus * tyy
                Tyz[iobs, icell] = neg_sus * tyz
                Tzx[iobs, icell] = neg_sus * tzx
                Tzy[iobs, icell] = neg_sus * tzy
                Tzz[iobs, icell] = neg_sus * tzz

    if ti.static(not compressed):
        if ti.static(write_to_mat):
            for i in range(3 * nC):
                mat[i, i] += 1
        else:
            for i in range(nC):
                Txx[i, i] += 1
                Tyy[i, i] += 1
                Tzz[i, i] += 1
