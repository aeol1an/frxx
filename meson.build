project(
  'frxx',
  'cpp',
  version: '0.1.4.2',
  license: 'LGPL-3.0-or-later',
  meson_version: '>=1.1.0',
  default_options: ['cpp_std=c++17'],
)

# Use platlib for both the Python sources and the forthcoming C++ extension so
# meson-python can place the complete package in a single wheel install scheme.
py = import('python').find_installation(pure: false)
pybind11_dep = dependency('pybind11')
threads_dep = dependency('threads')
frxx_pybind_inc = include_directories('src')

subdir('src')
