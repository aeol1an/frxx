project(
  'frxx',
  'cpp',
  version: '0.1.2',
  license: 'LGPL-3.0-or-later',
  meson_version: '>=1.1.0',
  default_options: ['cpp_std=c++17'],
)

# Use platlib for both the Python sources and the forthcoming C++ extension so
# meson-python can place the complete package in a single wheel install scheme.
py = import('python').find_installation(pure: false)
pybind11_dep = dependency('pybind11')

install_subdir(
  'src/frxx',
  install_dir: py.get_install_dir(),
  exclude_directories: [
    '__pycache__',
    'core/__pycache__',
    'io/__pycache__',
    'io/decoders/__pycache__',
    'io/decoders/rk/__pycache__',
    'misc/__pycache__',
    'proc/__pycache__',
    'proc/algs/__pycache__',
    'proc/moments/__pycache__',
    'proc/spectra/__pycache__',
    'utils/__pycache__',
    'viz/__pycache__',
  ],
  install_tag: 'runtime',
)
