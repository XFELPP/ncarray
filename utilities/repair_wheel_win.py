#!python
import os
import sys
import glob
import subprocess
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
        cuda_path = os.path.normpath(cuda_path)
        builtins_dlls: List[str] = glob.glob(f"{cuda_path}/bin/nvrtc-builtins64_*.dll")

        if not builtins_dlls:
            print("No NVRTC builtins dlls found!")
        else:
            selected_dll: str = os.path.basename(builtins_dlls[0])
            delvewheel_search_path = f"{delvewheel_search_path};{cuda_path}/bin"

            cuda_args = [
                "--add-dll",
                selected_dll,
                "--no-dll",
                "nvcuda.dll",
            ]

    delvewheel_cmd.extend(["--add-path", delvewheel_search_path])
    delvewheel_cmd.extend(cuda_args)
    delvewheel_cmd.extend(["-w", dest_dir, wheel_path])

    subprocess.run(delvewheel_cmd, check=True)


if __name__ == "__main__":
    main()
