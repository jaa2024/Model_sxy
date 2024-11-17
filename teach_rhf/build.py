import os
import subprocess
import ctypes
import shutil
import platform

# 设置库的路径
lib_dir = "libcint/build"

# 判断系统类型来设置库文件名
system = platform.system()
if system == "Linux":
    lib_name = "libcint.so"
elif system == "Darwin":  # macOS
    lib_name = "libcint.dylib"
else:
    raise OSError("Unsupported operating system")

lib_path = os.path.join(lib_dir, lib_name)


def build_lib():
    # 检查库文件是否存在
    if not os.path.exists(lib_path):
        print("Library not found, building libcint...")

        # 如果 build 目录已存在，先删除
        if os.path.exists(lib_dir):
            print("Build directory exists, removing...")
            shutil.rmtree(lib_dir)

        # 创建新的 build 目录并执行 CMake 和 Make
        os.makedirs(lib_dir, exist_ok=True)
        subprocess.check_call(
            [
                "cmake",
                "-DWITH_RANGE_COULOMB=ON",
                "-DENABLE_EXAMPLE=0",
                "-DENABLE_TEST=0",
                "-DWITH_RANGE_COULOMB=1",
                "-DWITH_F12=1",
                "-DWITH_COULOMB_ERF=1",
                "-DMIN_EXPCUTOFF=20",
                "-DKEEP_GOING=1",
                "..",
            ],
            cwd=lib_dir,
        )
        subprocess.check_call(["make", "-j"], cwd=lib_dir)
        print("Library built successfully.")
    else:
        print("Library already exists, skipping build.")

    # 加载编译好的动态库
    libcint = ctypes.cdll.LoadLibrary(lib_path)
    return libcint


# 使用示例
if __name__ == "__main__":
    libcint = build_lib()
    # 设置函数的 argtypes 和 restype
    # 示例：libcint.some_function.argtypes = [ctypes.c_int, ctypes.c_double]
    # 示例：libcint.some_function.restype = ctypes.c_double
