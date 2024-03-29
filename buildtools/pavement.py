"""
用于生成Release commit以及changelog和软件包
除了requirements-dev以外还需要环境中存在git，nodejs以及定制的gitmoji-changelog（以子模块形式存在）

如果你在clone本仓库时没有使用--recursive来初始化子模块，那么请先用如下命令拉取子模块
    git submodule update --init --recursive

如果没有安装yarn的话可能还需要安装yarn
    npm install -g yarn

然后使用yarn安装gitmoji-changelog所需依赖（假设你当前在项目根目录）
    cd ./buildtools/gitmoji-changelog & yarn

环境准备完成！

基础用法：
    python ./buildtools/pavement.py -u -w -r {version} -c -b
    其中-u代表如果当前HEAD属于{version}且对changelog进行了更改，则更新release commit，并阻止其他操作
       -w代表如果当前HEAD属于{version}，则撤销当前release commit
       -r代表指定发布版本号为{version}
       -c代表生成release commit
       -b代表进行生成打包

    python ./buildtools/pavement.py -w
    其中不指定-r参数时，-w代表撤销当前HEAD对应的release commit，如果HEAD不为release commit则跳过

    python ./buildtools/pavement.py
    直接生成changelog，版本取决于当前HEAD

    python ./buildtools/pavement.py -b
    直接进行生成打包（不生成changelog），版本取决于当前HEAD

工作流：
    1. 执行该命令生成版本release commit
    python ./buildtools/pavement.py -u -w -r {version} -c

    2. 编辑changelog或者release-notes

    3. 再次执行该命令将修改后的日志文件更新到release commit中
    python ./buildtools/pavement.py -u -w -r {version} -c

    4. (可选) 日志有错误，执行该命令撤销release commit
    python ./buildtools/pavement.py -w

    5. 然后重复步骤1-3生成新的release commit（**警告**：之前第2步做出的编辑会被覆盖）

    6. 完成release commit生成后执行打包操作
    python ./buildtools/pavement.py -b
"""

import argparse
import contextlib
import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

import tomli


class Pavement:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.description = 'Generate changelog, make a tagged commit and build the package.'
        parser.add_argument('-r', '--release',
                            help='Specify the release version code.\n'
                                 'If not specified, version code will be generated based on repo status.',
                            type=str, default=None)
        parser.add_argument('-c', '--commit',
                            help='Make a release commit based on changelog and tag it with version code.',
                            action='store_true')
        parser.add_argument('-u', '--update',
                            help='Update release commit based on new changes to changelogs and re-tag it.'
                                 'If an update is executed,'
                                 'then prevent changelog generation and withdraw operation if enabled.',
                            action='store_true')
        parser.add_argument('-w', '--withdraw',
                            help='Revert the release tag and corresponding commit.'
                                 'If a withdrawal is executed, the prevent changelog generation.',
                            action='store_true')
        parser.add_argument('-b', '--build',
                            help='Build the package.',
                            action='store_true')
        parser.add_argument('-p', '--path',
                            help='Path to the repository. Default is current work directory',
                            type=str, default=None)
        args = parser.parse_args()

        self.pavement_path = Path(os.path.realpath(sys.argv[0])).resolve()
        if args.path is not None:
            self.repo_path = repo_path = Path(args.path).resolve()
            os.chdir(args.path)
        else:
            self.repo_path = repo_path = Path().resolve()

        properties = tomli.loads(Path('pyproject.toml').read_text())
        self.description = properties['project']['description']
        self.name = properties['project']['name']

        self.is_release = False
        if args.release is not None:
            self.version = args.release
            self.is_release = True

        self.commit = args.commit
        self.update = args.update
        self.withdraw = args.withdraw
        self.build = args.build

        # 指示是否只进行build only，该情况下不会更新readme
        self.build_only = self.build and not self.commit and not self.update and not self.withdraw

        self.changelog_file = repo_path / f'docs/changelog/{self.version}-changelog.md'
        self.release_notes_file = repo_path / f'docs/release/{self.version}-notes.md'
        self.complete_changelog_file = repo_path / f'docs/changelog/CHANGELOG-ALL.md'

        self._head_description = None

    @property
    def version(self):
        """根据当前仓库状态和命令行参数可能存在三种状态
        1. 命令行使用--release或-r指定发布版本，则此时version与指定的值一致
        2. 命令行未指定，但当前仓库的HEAD是release commit，则version与当前仓库release commit指定的版本一致
        3. 命令行未指定且当前仓库的HEAD不是release commit，则version为上一个版本加上"+post{d}"后缀，其中{d}为到上一次发布的commit距离
        """
        if getattr(self, '_version', None) is None:
            if self.is_head_release_commit():
                self.version = self.head_description
            else:
                sys.path.insert(0, self.repo_path.__fspath__())
                version = importlib.import_module(f'{self.name}.version')
                _version = version.short_version
                if version.post_distance > 0:
                    _version = f'{_version}+post{version.post_distance}'
                self.version = _version

        return self._version

    @version.setter
    def version(self, value):
        """version的setter会去掉传入version字符串的v前缀，以适用于version_tag属性
        """
        if value is None:
            return
        self._version = value
        if self._version.startswith('v'):
            self._version = self.version[1:]

    @property
    def version_tag(self):
        """version_tag特指以"v"开头的版本号
        """
        return f'v{self.version}'

    @property
    def head_description(self):
        """获取git describe的结果
        """
        if getattr(self, '_head_description', None) is None:
            # 假定version tag里不包含非ascii字符
            self._head_description = subprocess.check_output(['git', 'describe'], **{'text': 'utf-8'}).strip()

        return self._head_description

    def is_head_release_commit(self):
        """判断当前HEAD是否为release commit
        """
        return re.compile(r'-\d+-\w{8}$').search(self.head_description) is None

    def is_head_current_release_commit(self):
        """判断当前HEAD是否就是--release或-r指定的版本release commit
        """
        return self.head_description == self.version_tag

    def delete_tag(self):
        subprocess.check_call(['git', 'tag', '-d', f'{self.version_tag}'])

    def make_tag(self):
        subprocess.check_call(['git', 'tag', '-a', f'{self.version_tag}', '-m', f'Release {self.version_tag}'])

    def add_release_files(self, command=('add',)):
        subprocess.check_call(['git', *command, f'{self.changelog_file}'])
        subprocess.check_call(['git', *command, f'{self.release_notes_file}'])

    def checkout_release_files_from_stash(self):
        self.add_release_files(command=(
            'checkout',
            'stash@{0}',
            '--'
        ))

    def generate_changelog(self):
        """依靠gitmoji-changelog生成changelog，需要额外安装npm和gitmoji-changelog
        参考：
        https://docs.gitmoji-changelog.dev/#/?id=generic
        """
        repo_path = self.repo_path
        changelog_config_file = repo_path / '.gitmoji-changelogrc'
        content = f"""{{
  "project": {{
    "name": "{self.name}",
    "description": "{self.description}",
    "version": "{self.version}"
  }}
}}"""
        changelog_config_file.write_text(content)
        complete_changelog_file = self.complete_changelog_file
        changelog_file = self.changelog_file
        release_notes_file = self.release_notes_file

        output_file = complete_changelog_file.absolute()

        if changelog_file.exists():
            changelog_file.unlink()

        # TODO: node执行文件用subprocess函数会提示找不到文件
        changelog_script_path = 'gitmoji-changelog/packages/gitmoji-changelog-cli/src/index.js'
        subprocess.check_call(['node', f'{self.pavement_path.parent / changelog_script_path}',
                               '--preset', 'generic',
                               '--output', f'{output_file}'])

        changelog_config_file.unlink()

        # 由于目前gitmoji-changelog会输出完整changelog，因此需要将最新的changelog分割出来
        changelog_all = complete_changelog_file.read_text(encoding='utf-8')
        changelog_for_current_version = re.compile('<a name=".*?"></a>').split(changelog_all)[1]
        changelog_file.write_text(changelog_for_current_version, encoding='utf-8')

        rn_repl = f"""## METALpy \\1 Release Notes

发布 METALpy {self.version}."""

        release_notes_for_current_version = re.compile('^## (.*?)( .*?)?$', re.M) \
            .sub(rn_repl, changelog_for_current_version, 1)
        release_notes_file.write_text(release_notes_for_current_version, encoding='utf-8')

    def build_package(self):
        os.system(f'{sys.executable} -m build {self.repo_path}')

    def make_release_commit(self):
        with self.git_stashed():
            self.add_release_files()
            subprocess.check_call(['git', 'commit', '-m', f':bookmark: 发布{self.version}'])
            self.make_tag()

    def update_release_commit(self):
        # 需要当前的HEAD就是release commit才能update
        if not self.is_head_release_commit():
            print('HEAD is not the release commit, skip updating.')
            return False

        if self.is_release and not self.is_head_current_release_commit():
            print("HEAD does not match the version being releasing, skip updating.")
            return False

        try:
            with self.git_stashed():
                self.checkout_release_files_from_stash()  # 从stash中checkout出release commit文件
                self.add_release_files()
                self.delete_tag()
                subprocess.check_call(['git', 'commit', '--amend', '--no-edit'])
                self.make_tag()
        except subprocess.CalledProcessError:
            return False

        return True

    def withdraw_changes(self):
        # 需要当前的HEAD就是release commit才能withdraw
        # 否则describe结果会形如v0.0.5-1-ga5b0357
        if not self.is_head_release_commit():
            print('HEAD is not the release commit, skip withdrawing.')
            return False

        if self.is_release and not self.is_head_current_release_commit():
            print("HEAD does not match the version being releasing, skip withdrawing.")
            return False

        with self.git_stashed():
            self.delete_tag()
            subprocess.check_call(['git', 'reset', '--mixed', 'HEAD~1'])

        return True

    @contextlib.contextmanager
    def git_stashed(self):
        stash_name = f"Before {self.version}"
        output = subprocess.check_output(['git', 'stash', 'save', stash_name], **{'text': 'utf-8'})

        # 判断是否成功stash，如果未成功，则清理步骤也不执行还原操作
        is_stashed = False
        if stash_name in output:
            # 因为指定了stash的名字，所以如果成功stash，输出中会包含stash的名字
            is_stashed = True

        try:
            yield
        finally:
            if is_stashed:
                subprocess.check_call(['git', 'stash', 'pop'])

    def main(self):
        withdraw = self.withdraw
        commit = self.commit
        if self.update:
            # 如果成功update，则不进行commit
            update_failed = not self.update_release_commit()
            commit = commit and update_failed
            withdraw = withdraw and update_failed
        else:
            update_failed = True

        withdraw_failed = True
        if withdraw:
            withdraw_failed = not self.withdraw_changes()

        if not self.build_only and update_failed and withdraw_failed:
            # 如果update和withdraw均失败或未进行，则生成changelog
            # 特例：如果只进行build操作，则不生成changelog
            self.generate_changelog()

        if commit:
            self.make_release_commit()

        if self.build:
            with self.git_stashed():
                self.build_package()


if __name__ == '__main__':
    Pavement().main()
