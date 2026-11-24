# Modified by Axolotl AI, 2026. Changes are listed in vendor/sources.json.
import os
import shutil
import setuptools
import torch
from setuptools import find_packages
from setuptools.command.build_py import build_py
from packaging.version import parse
from pathlib import Path
from torch.utils.cpp_extension import CUDAExtension, CUDA_HOME
from scripts.generate_pyi import generate_pyi_file


DG_SKIP_CUDA_BUILD = int(os.getenv('DG_SKIP_CUDA_BUILD', '0')) == 1

# Compiler flags
cxx_flags = ['-std=c++20', '-O3', '-fPIC', '-Wno-psabi', '-Wno-deprecated-declarations',
             f'-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}']

# Sources
current_dir = os.path.dirname(os.path.realpath(__file__))
sources = ['csrc/python_api.cpp']
build_include_dirs = [
    f'{CUDA_HOME}/include',
    f'{CUDA_HOME}/include/cccl',
    'optmoe_deepgemm/include',
    'third-party/deep_jit/include',
    'third-party/cutlass/include',
]
build_libraries = ['cudart']
build_library_dirs = [f'{CUDA_HOME}/lib64']
third_party_include_dirs = [
    'third-party/cutlass/include/cute',
    'third-party/cutlass/include/cutlass',
]

# Headers python_api.cpp can include (CUTLASS left out: it changes only with a pin bump): distutils rebuilds
# optmoe_deepgemm._C only when a source or one of these is newer than the built extension, so a header-only
# edit would otherwise reuse a stale object
depends = sorted(
    str(path.relative_to(current_dir))
    for root in ('csrc', 'optmoe_deepgemm/include', 'third-party/deep_jit/include')
    for pattern in ('*.hpp', '*.h', '*.cuh')
    for path in Path(current_dir, root).rglob(pattern)
)

def get_package_version():
    """``OPTMOE_DEEPGEMM_VERSION`` (the optmoe release this build belongs to), tagged with the CUDA
    major and torch minor version the extension links against."""
    public_version = os.getenv('OPTMOE_DEEPGEMM_VERSION', '0.0.0.dev0').split('+')[0]
    cuda_version = parse(torch.version.cuda).major
    torch_version = parse(torch.__version__)
    return f'{public_version}+cu{cuda_version}.torch{torch_version.major}.{torch_version.minor}'


def get_ext_modules():
    if DG_SKIP_CUDA_BUILD:
        return []

    return [CUDAExtension(name='optmoe_deepgemm._C',
                          sources=sources,
                          depends=depends,
                          include_dirs=build_include_dirs,
                          libraries=build_libraries,
                          library_dirs=build_library_dirs,
                          extra_compile_args=cxx_flags)]


class CustomBuildPy(build_py):
    def run(self):
        # First, prepare the include directories
        self.prepare_includes()

        # Second, make clusters' cache setting default into `envs.py`
        self.generate_default_envs()

        # Third, generate and copy .pyi file to build root directory
        self.generate_pyi_file()

        # Fourth, copy csrc and docs into the wheel for agent-side error lookup
        self.prepare_agent_files()

        # Finally, run the regular build
        build_py.run(self)

    def prepare_agent_files(self):
        # Copy csrc and docs into the wheel for agent-side error lookup
        package_dir = os.path.join(self.build_lib, 'optmoe_deepgemm')
        for name in ('csrc', 'docs'):
            dst = os.path.join(package_dir, name)
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(os.path.join(current_dir, name), dst)

    def generate_pyi_file(self):
        generate_pyi_file(name='_C', root='./csrc', output_dir='./stubs')
        pyi_source = os.path.join(current_dir, 'stubs', '_C.pyi')
        pyi_target = os.path.join(self.build_lib, 'optmoe_deepgemm', '_C.pyi')

        if os.path.exists(pyi_source):
            print(f"Copying .pyi file from {pyi_source} to {pyi_target}")
            os.makedirs(os.path.dirname(pyi_target), exist_ok=True)
            shutil.copy2(pyi_source, pyi_target)
        else:
            print(f"Warning: .pyi file not found at {pyi_source}")

    def generate_default_envs(self):
        code = '# Pre-installed environment variables\n'
        code += 'persistent_envs = dict()\n'
        for name in ('DG_JIT_CACHE_DIR', 'DG_JIT_PRINT_COMPILER_COMMAND', 'DG_JIT_CPP_STANDARD'):
            code += f"persistent_envs['{name}'] = '{os.environ[name]}'\n" if name in os.environ else ''

        with open(os.path.join(self.build_lib, 'optmoe_deepgemm', 'envs.py'), 'w') as f:
            f.write(code)

    def prepare_includes(self):
        # Create temporary build directory instead of modifying package directory
        build_include_dir = os.path.join(self.build_lib, 'optmoe_deepgemm/include')
        os.makedirs(build_include_dir, exist_ok=True)

        # Copy third-party includes to the build directory
        for d in third_party_include_dirs:
            dirname = d.split('/')[-1]
            src_dir = os.path.join(current_dir, d)
            dst_dir = os.path.join(build_include_dir, dirname)

            # Remove existing directory if it exists
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)

            # Copy the directory
            shutil.copytree(src_dir, dst_dir)


if __name__ == '__main__':
    # noinspection PyTypeChecker
    setuptools.setup(
        name='optmoe-deepgemm',
        version=get_package_version(),
        packages=find_packages('.'),
        package_data={
            'optmoe_deepgemm': [
                'include/deep_gemm/**/*',
                'include/cute/**/*',
                'include/cutlass/**/*',
            ]
        },
        ext_modules=get_ext_modules(),
        zip_safe=False,
        cmdclass={
            'build_py': CustomBuildPy,
        },
    )
