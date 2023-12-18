import copy

from metalpy.utils.dhash import dhash
from metalpy.utils.type import Self
from . import CompositeTransform, Transform, Translation, Rotation


class Transformable:
    def __init__(self):
        self.transforms = CompositeTransform()

    def apply_transform(self, trans: Transform, inplace=False) -> Self:
        """逻辑上对空间体位置进行变换，目前通过对网格点进行逆变换实现

        Parameters
        ----------
        trans
            待应用的变换
        inplace
            指示操作是否应用在当前实例上

        Returns
        -------
        ret
            返回应用变换后的对象，如果 `inplace=True` 则返回当前对象，否则返回拷贝对象
        """
        if inplace:
            ret = self
        else:
            ret = copy.deepcopy(self)

        ret.transforms.add(trans)

        return ret

    def translate(self, x, y, z, inplace=False) -> Self:
        """对Shape进行平移

        Parameters
        ----------
        x, y, z
            三个方向的位移量
        inplace
            指示操作是否应用在当前实例上，默认为 `False`

        Returns
        -------
        ret
            返回变换后的对象，如果 `inplace=True` 则返回当前对象，否则返回拷贝对象
        """
        return self.apply_transform(Translation(x, y, z), inplace=inplace)

    def translated(self, x, y, z) -> Self:
        """对Shape进行平移

        同 `translate` ， 但默认启用 `inplace`

        Parameters
        ----------
        x, y, z
            三个方向的位移量

        Returns
        -------
        ret
            返回变换后的当前对象
        """
        return self.translate(x, y, z, inplace=True)

    def rotate(self, a, b, y, degrees=True, radians=False, seq='xyz', inplace=False) -> Self:
        """对Shape进行旋转，方向遵循右手准则

        Parameters
        ----------
        a, b, y
            分别对应于x、y、z方向的旋转
        degrees
            指示y，a，b是否为角度制，默认为True，即默认使用角度制，否则使用弧度制
        radians
            指示y，a，b是否为弧度制，默认为False。如果为True，则忽略 `degrees` 参数，使用弧度制
        seq
            旋转顺序，决定旋转顺序和内外旋，参考 :meth:`scipy.spatial.transform.Rotation.from_euler`
        inplace
            指示操作是否应用在当前实例上，默认为 `False`

        Returns
        -------
        ret
            返回变换后的对象，如果 `inplace=True` 则返回当前对象，否则返回拷贝对象
        """
        return self.apply_transform(Rotation(a, b, y, degrees=degrees, radians=radians, seq=seq), inplace=inplace)

    def rotated(self, a, b, y, degrees=True, radians=False, seq='xyz') -> Self:
        """对Shape进行旋转，方向遵循右手准则

        同 `rotate` ， 但默认启用 `inplace`

        Parameters
        ----------
        a, b, y
            分别对应于x、y、z方向的旋转
        degrees
            指示y，a，b是否为角度制，默认为True，即默认使用角度制，否则使用弧度制
        radians
            指示y，a，b是否为弧度制，默认为False。如果为True，则忽略 `degrees` 参数，使用弧度制
        seq
            旋转顺序，决定旋转顺序和内外旋，参考 :meth:`scipy.spatial.transform.Rotation.from_euler`

        Returns
        -------
        ret
            返回变换后的当前对象
        """
        return self.rotate(a, b, y, degrees=degrees, radians=radians, seq=seq, inplace=True)

    def rotate_x(self, angle, degrees=True, radians=False, inplace=False) -> Self:
        """对Shape的x方向进行旋转，方向遵循右手准则

        Parameters
        ----------
        angle
            旋转量
        degrees
            指示是否为角度制，默认为True，即默认使用角度制，否则使用弧度制
        radians
            指示y，a，b是否为弧度制，默认为False。如果为True，则忽略 `degrees` 参数，使用弧度制
        inplace
            指示操作是否应用在当前实例上

        Returns
        -------
        ret
            返回变换后的对象，如果 `inplace=True` 则返回当前对象，否则返回拷贝对象
        """
        return self.rotate(angle, 0, 0, degrees=degrees, radians=radians, inplace=inplace)

    def rotate_y(self, angle, degrees=True, radians=False, inplace=False) -> Self:
        """对Shape的y方向进行旋转，方向遵循右手准则

        Parameters
        ----------
        angle
            旋转量
        degrees
            指示是否为角度制，默认为True，即默认使用角度制，否则使用弧度制
        radians
            指示y，a，b是否为弧度制，默认为False。如果为True，则忽略 `degrees` 参数，使用弧度制
        inplace
            指示操作是否应用在当前实例上

        Returns
        -------
        ret
            返回变换后的对象，如果 `inplace=True` 则返回当前对象，否则返回拷贝对象
        """
        return self.rotate(0, angle, 0, degrees=degrees, radians=radians, inplace=inplace)

    def rotate_z(self, angle, degrees=True, radians=False, inplace=False) -> Self:
        """对Shape的z方向进行旋转，方向遵循右手准则

        Parameters
        ----------
        angle
            旋转量
        degrees
            指示是否为角度制，默认为True，即默认使用角度制，否则使用弧度制
        radians
            指示y，a，b是否为弧度制，默认为False。如果为True，则忽略 `degrees` 参数，使用弧度制
        inplace
            指示操作是否应用在当前实例上

        Returns
        -------
        ret
            返回变换后的对象，如果 `inplace=True` 则返回当前对象，否则返回拷贝对象
        """
        return self.rotate(0, 0, angle, degrees=degrees, radians=radians, inplace=inplace)

    def __dhash__(self):
        return dhash(self.transforms)
