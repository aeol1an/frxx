cpp_sources = []

subdir('frxx')

frxx_cpp_lib = static_library(
    'frxx_cpp',
    cpp_sources,
    include_directories: frxx_cpp_inc,
    dependencies: threads_dep,
    install: false,
    pic: true,
)

frxx_dep = declare_dependency(
    include_directories: frxx_cpp_inc,
    dependencies: threads_dep,
    link_with: frxx_cpp_lib,
)
