python_sources = files(
    '__init__.py',
    '_backend.py',
    'cli.py'
)

py.install_sources(
    python_sources,
    subdir: 'frxx',
    install_tag: 'runtime'
)

subdir('core')
subdir('io')
subdir('proc')
subdir('utils')
subdir('viz')
