python_sources = files(
    '__init__.py',
    'ACF.py',
    'bootstrapDPSD.py',
    'fuzzyDCA.py',
    'res.py'
)

py.extension_module(
    '_res',
    'res.cpp',
    dependencies: [pybind11_dep, frxx_dep],
    subdir: 'frxx/proc/algs',
    install: true,
    install_tag: 'runtime'
)

py.install_sources(
    python_sources,
    subdir: 'frxx/proc/algs',
    install_tag: 'runtime'
)
