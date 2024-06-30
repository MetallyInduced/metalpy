__version__, __version_tuple__ = '__UNSPECIFIED__', '__UNSPECIFIED__'


def get_version():
    from pathlib import Path
    error = ImportError

    if __version__ == '__UNSPECIFIED__':
        try:
            from versioningit import get_version, NotVersioningitError, NotSdistError, NotVCSError
            error = NotVersioningitError, NotSdistError, NotVCSError

            return get_version(Path(__file__).parent.parent)
        except error:
            return '999.999.999'
    else:
        return __version__


def get_version_tuple():
    from pathlib import Path
    error = ImportError

    if __version_tuple__ == '__UNSPECIFIED__':
        try:
            from versioningit import Versioningit, NotVersioningitError, NotVCSError, NotSdistError
            error = NotVersioningitError, NotVCSError, NotSdistError

            version_tuple_str = Versioningit\
                .from_project_dir(Path(__file__).parent.parent)\
                .do_template_fields(get_version(), None, None, None)['version_tuple']

            return eval(version_tuple_str)
        except error:
            return 999, 999, 999, '', ''
    else:
        return __version_tuple__
