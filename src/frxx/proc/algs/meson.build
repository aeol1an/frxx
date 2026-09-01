python_sources = files(
    '__init__.py',
    'ACF.py',
    'bootstrapDPSD.py',
    'fuzzyDCA.py',
    'res.py'
)

py.extension_module(
    '_ACF',
    'ACF.cpp',
    dependencies: [pybind11_dep, frxx_dep],
    include_directories: frxx_pybind_inc,
    subdir: 'frxx/proc/algs',
    install: true,
    install_tag: 'runtime'
)

py.extension_module(
    '_bootstrapDPSD',
    'bootstrapDPSD.cpp',
    dependencies: [pybind11_dep, frxx_dep],
    include_directories: frxx_pybind_inc,
    subdir: 'frxx/proc/algs',
    install: true,
    install_tag: 'runtime'
)

py.extension_module(
    '_res',
    'res.cpp',
    dependencies: [pybind11_dep, frxx_dep],
    include_directories: frxx_pybind_inc,
    subdir: 'frxx/proc/algs',
    install: true,
    install_tag: 'runtime'
)

py.extension_module(
    '_fuzzyDCA',
    'fuzzyDCA.cpp',
    dependencies: [pybind11_dep, frxx_dep],
    include_directories: frxx_pybind_inc,
    subdir: 'frxx/proc/algs',
    install: true,
    install_tag: 'runtime'
)

py.install_sources(
    python_sources,
    subdir: 'frxx/proc/algs',
    install_tag: 'runtime'
)
