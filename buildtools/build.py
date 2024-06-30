import os

import nox

PYTHONS = ['3.9']


def find_wheel():
    path = './dist'

    if not os.path.exists(path):
        return None

    for file in os.listdir(path):
        if file.endswith('.whl'):
            return os.path.join(path, file)
    else:
        return None


@nox.session(python=PYTHONS)
def build_wheel(session: nox.Session) -> None:
    session.chdir('..')
    session.install('-e', '.[build]')
    session.run('python', './buildtools/pavement.py', '-b')


@nox.session(python=PYTHONS)
def test(session: nox.Session) -> None:
    session.chdir('..')
    session.install('pytest')

    wheel = find_wheel()

    if wheel is not None and (session.posargs + [None])[0] == 'refs/heads/main':
        # 在主分支上进行打包测试
        session.install(wheel + '[complete]')
        session.chdir('dist')
        session.run('pytest', '--pyargs', 'metalpy')
    else:
        # 其他分支上直接进行测试
        session.install('-e', '.[complete]')
        session.run('pytest', '.')
