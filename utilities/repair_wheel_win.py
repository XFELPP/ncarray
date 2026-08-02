#!python
import os
import sys
import glob
import subprocess
import zipfile
from typing import List, Optional


def main():
    if len(sys.argv) < 3:
        print("Usage: repair_wheel.py <dest_dir> <wheel_path>")
        sys.exit(1)

    dest_dir: str = sys.argv[1]
    wheel_path: str = sys.argv[2]

    # Check if the exclusion flag is set. When building in the split mode you
    # do NOT exclude them, otherwise you do.
    no_exclude_core: bool = "--no-exclude-core" in sys.argv

    scripts_dir: str = os.path.dirname(os.path.abspath(__file__))
    root_dir: str = os.path.dirname(scripts_dir)
    bin_dir: str = os.path.normpath(os.path.join(root_dir, "install", "bin"))
    lib_dir: str = os.path.normpath(os.path.join(root_dir, "install", "lib"))

    delvewheel_cmd: List[str] = ["delvewheel", "repair", "--no-dll", "eagereval-1.dll"]
    if not no_exclude_core:
        delvewheel_cmd.extend(
            [
                "--no-dll",
                "ncarray-1.dll",
                "--no-dll",
                "ncdevarray-1.dll",
                "--no-dll",
                "ncarrayjit-1.dll",
            ]
        )

    delvewheel_search_path: str = f"{lib_dir};{bin_dir}"

    cuda_args: List[str] = []
    cuda_path: Optional[str] = os.getenv("CUDA_PATH")
    if cuda_path is None:
        print(
            "CUDA_PATH is not defined! Cannot find CUDA! Skipping NVRTC builtins packaging."
        )
    else:
        cuda_args.extend(["--no-dll", "nvcuda.dll"])

        cuda_path = os.path.normpath(cuda_path)
        # Seems like there's an [ARCH] sub-directory after CUDA13....
        # See: https://github.com/shader-slang/slangpy/issues/614
        cuda_dlls: List[str] = glob.glob(f"{cuda_path}/**/*.dll", recursive=True)
        cuda_dlls_dirs: List[str] = sorted(
            list(set(os.path.dirname(os.path.normpath(p)) for p in cuda_dlls))
        )
        for dll_dir in cuda_dlls_dirs:
            delvewheel_search_path = f"{delvewheel_search_path};{dll_dir}"

        builtins_dlls: List[str] = glob.glob(
            f"{cuda_path}/**/nvrtc-builtins*.dll", recursive=True
        )

        if not builtins_dlls:
            print("No NVRTC builtins dlls found!")
        else:
            if len(builtins_dlls) > 1:
                print(f"Multiple NVRTC DLLs found: {builtins_dlls}")
            selected_dll: str = os.path.basename(builtins_dlls[0])

            cuda_args.extend(["--add-dll", selected_dll])

    delvewheel_cmd.extend(["--add-path", delvewheel_search_path])
    delvewheel_cmd.extend(cuda_args)
    delvewheel_cmd.extend(["-w", dest_dir, wheel_path])

    subprocess.run(delvewheel_cmd, check=True)

    # Add back `.lib` files.... How on erth does Windows build system work!??!?
    repaired_wheels: List[str] = glob.glob(os.path.join(dest_dir, "*.whl"))
    for whl in repaired_wheels:
        print(f"Injecting .lib import libraries into {whl}...")
        with zipfile.ZipFile(whl, "a") as z:
            for lib_file in glob.glob(os.path.join(lib_dir, "*.lib")):
                target_path: str = f"ncarray/lib/{os.path.basename(lib_file)}"
                print(f"  Adding {lib_file} -> {target_path}")
                z.write(lib_file, target_path)


if __name__ == "__main__":
    main()
