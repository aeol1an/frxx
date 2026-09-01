python_sources = files(
    '__init__.py',
    'fuzzyDCA.py',
    'standard.py'
)

py.extension_module(
    '_standard',
    'standard.cpp',
    dependencies: [pybind11_dep, frxx_dep],
    include_directories: frxx_pybind_inc,
    subdir: 'frxx/proc/moments',
    install: true,
    install_tag: 'runtime'
)

py.install_sources(
    python_sources,
    subdir: 'frxx/proc/moments',
    install_tag: 'runtime'
)
