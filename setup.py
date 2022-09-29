import distutils.cmd
import distutils.log
import errno
import subprocess
import os
from setuptools import setup


class CudaCommand(distutils.cmd.Command):
    description = 'Compile CUDA sources'
    user_options = [
        ('gcc=', None, 'Path to the gcc compiler.'),
        ('gcc-4-8=', None, 'Path to gcc-4.8 compiler.'),
        ('nvcc=', None, 'Path to the Nvidia nvcc compiler.'),
        ('cuda-lib=', None, 'Path to the CUDA libraries.'),
    ]

    def run(self):
        src_path = os.path.abspath('./CUDAsrc')
        build_path = os.path.abspath('./CUDAbuild')
        print('Source Path:\t' + src_path)
        print('Build Path:\t' + build_path)

        import tensorflow as tf
        tf_cflags = tf.sysconfig.get_compile_flags()
        tf_lflags = tf.sysconfig.get_link_flags()
        link_cuda_lib = '-L' + self.cuda_lib

        # Remove old build files
        if os.path.isdir(build_path):
            print('Removing old build files from %s' % build_path)
            for file in os.listdir(build_path):
                os.remove(os.path.join(build_path, file))
        else:
            print('Creating build directory at %s' % build_path)
            os.mkdir(build_path)

        print('Compiling CUDA code...')

        print('LinearSolve -- Multi BiCgStab iLU')
        subprocess.check_call(
            [
                self.nvcc,
                '-std=c++11',
                '-c',
                '-o',
                os.path.join(build_path, 'multi_bicgstab_ilu_linear_solve_op.cu.o'),
                os.path.join(src_path, 'multi_bicgstab_ilu_linear_solve_op.cu.cc'),
                # '-O2',
                '-x',
                'cu',
                '-O3',
                '-Xptxas',
                '-O3',
                '-Xcompiler',
                '-fPIC'
            ]
            + tf_cflags
        )

        subprocess.check_call(
            [
                self.gcc_4_8,
                '-std=c++11',
                '-shared',
                '-o',
                os.path.join(build_path, 'multi_bicgstab_ilu_linear_solve_op.so'),
                os.path.join(src_path, 'multi_bicgstab_ilu_linear_solve_op.cc'),
                os.path.join(build_path, 'multi_bicgstab_ilu_linear_solve_op.cu.o'),
                '-fPIC'
            ]
            + tf_cflags
            + tf_lflags
            + ['-L/usr/local/cuda-10.0/lib64/', '-lcudart', '-lcublas', '-lcusolver', '-lcusparse']
        )

        print('LinearSolve -- BiCgStab iLU')

        subprocess.check_call(
            [
                self.nvcc,
                '-std=c++11',
                '-c',
                '-o',
                os.path.join(build_path, 'bicgstab_ilu_linear_solve_op.cu.o'),
                os.path.join(src_path, 'bicgstab_ilu_linear_solve_op.cu.cc'),
                #'-O2',
                '-x',
                'cu',
                '-O3',
                '-Xptxas',
                '-O3',
                '-Xcompiler',
                '-fPIC'
            ]
            + tf_cflags
        )

        subprocess.check_call(
            [
                self.gcc_4_8,
                '-std=c++11',
                '-shared',
                '-o',
                os.path.join(build_path, 'bicgstab_ilu_linear_solve_op.so'),
                os.path.join(src_path, 'bicgstab_ilu_linear_solve_op.cc'),
                os.path.join(build_path, 'bicgstab_ilu_linear_solve_op.cu.o'),
                '-fPIC'
            ]
            + tf_cflags
            + tf_lflags
            + ['-L/usr/local/cuda-10.0/lib64/', '-lcudart', '-lcublas', '-lcusolver', '-lcusparse']
        )


        # Build the CentralDifferenceMatrix CUDA Kernels
        print('CentralDifference -- CSR matrix format')
        subprocess.check_call(
            [
                self.nvcc,
                '-std=c++11',
                '-c',
                '-o',
                os.path.join(build_path, 'central_difference_csr_op.cu.o'),
                os.path.join(src_path, 'central_difference_csr_op.cu.cc'),
                '-x',
                'cu',
                '-O3',
                '-Xptxas',
                '-O3',
                '-Xcompiler',
                '-fPIC',
                '-Xcompiler="-pthread"'
            ]
            + tf_cflags
        )
        subprocess.check_call(
            [
                self.gcc,
                '-std=c++11',
                '-shared',
                '-o',
                os.path.join(build_path, 'central_difference_csr_op.so'),
                os.path.join(src_path, 'central_difference_csr_op.cc'),
                os.path.join(build_path, 'central_difference_csr_op.cu.o'),
                '-fPIC'
            ]
            + tf_cflags
            + tf_lflags
            + ['-L/usr/local/cuda/lib64/', '-lcudart', '-lcublas', ]
        )

        # Build the Laplace Matrix Generation CUDA Kernels
        print('PressureSolve -- Laplace matrix op')
        subprocess.check_call(
            [
                self.nvcc,
                '-std=c++11',
                '-c',
                '-o',
                os.path.join(build_path, 'laplace_op.cu.o'),
                os.path.join(src_path, 'laplace_op.cu.cc'),
                '-x',
                'cu',
                '-O3',
                '-Xptxas',
                '-O3',
                '-Xcompiler',
                '-fPIC'
            ]
            + tf_cflags
        )

        # Build the Laplace Matrix Generation Custom Op
        # This is only needed for the Laplace Matrix Generation Benchmark
        subprocess.check_call(
            [
                self.gcc_4_8,
                '-std=c++11',
                '-shared',
                '-o',
                os.path.join(build_path, 'laplace_op.so'),
                os.path.join(src_path, 'laplace_op.cc'),
                os.path.join(build_path, 'laplace_op.cu.o'),
                '-fPIC'
            ]
            + tf_cflags
            + tf_lflags
            + ['-L/usr/local/cuda/lib64/','-lcudart']
        )

        # Build the Pressure Solver CUDA Kernels

        print('PressureSolve -- CG solver')
        subprocess.check_call(
            [
                self.nvcc,
                '-std=c++11',
                '-c',
                '-lcublas',
                '-lcurand',
                '-o',
                os.path.join(build_path, 'pressure_solve_op.cu.o'),
                os.path.join(src_path, 'pressure_solve_op.cu.cc'),
                '-x', 'cu',
                '-O3',
                '-Xptxas',
                '-O3',
                '-Xcompiler',
                '-fPIC'
            ]
            + tf_cflags
        )

        # Build the Pressure Solver Custom Op
        subprocess.check_call(
            [
                self.gcc_4_8,
                '-std=c++11',
                '-shared',
                '-o',
                os.path.join(build_path, 'pressure_solve_op.so'),
                os.path.join(src_path, 'pressure_solve_op.cc'),
                os.path.join(build_path, 'pressure_solve_op.cu.o'),
                os.path.join(build_path, 'laplace_op.cu.o'),
                '-fPIC'
            ]
            + tf_cflags
            + tf_lflags
            + ['-L/usr/local/cuda/lib64/','-lcudart']
        )

        print('Done with PISO files, heading into PhiFlow!')
        os.system('python PhiFlow/setup.py tf_cuda')

    def initialize_options(self):
        self.gcc = 'gcc'
        self.gcc_4_8 = 'g++-4.8'
        self.nvcc = 'nvcc'
        self.cuda_lib = '/usr/local/cuda/lib64/'


    def finalize_options(self):
        assert os.path.isfile(self.gcc) or self.gcc == 'gcc'
        assert os.path.isfile(self.nvcc) or self.nvcc == 'nvcc'


with open(os.path.join(os.path.dirname(__file__),'PhiFlow', 'phi', 'VERSION'), 'r') as version_file:
    version = version_file.read()


setup(
    cmdclass={
        'tf_cuda': CudaCommand,
    }
)
