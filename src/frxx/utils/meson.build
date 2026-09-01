python_sources = files(
    '__init__.py',
    'angleSplitter.py',
    'cfUtils.py',
    'coordConvert.py',
    'freqResolution.py',
    'cppHelpers.py',
    'windows.py',
    'pathUtils.py',
    'sourceFileUtils.py',
    'stringUtils.py'
)

py.install_sources(
    python_sources,
    subdir: 'frxx/utils',
    install_tag: 'runtime'
)

utils_extensions = {
    '_angleSplitter': 'angleSplitter.cpp',
    '_freqResolution': 'freqResolution.cpp',
    '_cppHelpers': 'cppHelpers.cpp',
    '_windows': 'windows.cpp',
}

foreach extension_name, extension_source : utils_extensions
    py.extension_module(
        extension_name,
        extension_source,
        dependencies: [pybind11_dep, frxx_dep],
        include_directories: frxx_pybind_inc,
        subdir: 'frxx/utils',
        install: true,
        install_tag: 'runtime'
    )
endforeach
