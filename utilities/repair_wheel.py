#!/usr/bin/env python3
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from typing import Dict, List, Tuple


def unmangle_wheel_libs_linux(
    whl_path: str,
    prefixes: Tuple[str, ...] = ("libncarray", "libncdevarray", "libncarrayjit"),
):
    tmp_dir: str = tempfile.mkdtemp(prefix="unmangle_whl_")
    try:
        with zipfile.ZipFile(whl_path, "r") as z:
            z.extractall(tmp_dir)

        rename_map: Dict[str, str] = {}
        for root, _, files in os.walk(tmp_dir):
            for f in files:
                if f.endswith(".so") or ".so." in f:
                    for p in prefixes:
                        if f.startswith(p):
                            clean_name: str = re.sub(r"-[0-9a-f]{8,}", "", f)
                            if clean_name != f:
                                old_path: str = os.path.join(root, f)
                                new_path: str = os.path.join(root, clean_name)
                                os.rename(old_path, new_path)
                                rename_map[f] = clean_name

                                subprocess.run(
                                    ["patchelf", "--set-soname", clean_name, new_path],
                                    check=False,
                                )
                            break
        if not rename_map:
            return

        for root, _, files in os.walk(tmp_dir):
            for f in files:
                if f.endswith(".so") or ".so." in f:
                    so_path: str = os.path.join(root, f)
                    for mangled_name, clean_name in rename_map.items():
                        subprocess.run(
                            [
                                "patchelf",
                                "--replace-needed",
                                mangled_name,
                                clean_name,
                                so_path,
                            ],
                            check=False,
                        )
        shutil.make_archive(whl_path.replace(".whl", ""), "zip", tmp_dir)
        os.replace(whl_path.replace(".whl", ".zip"), whl_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def update_pc_in_repaired_wheel(whl_path: str):
    with zipfile.ZipFile(whl_path, "r") as z:
        pc_path_in_whl: str = next(
            (f for f in z.namelist() if f.endswith("ncarray.pc")), ""
        )
        if not pc_path_in_whl:
            return

        pc_content: str = z.read(pc_path_in_whl).decode("utf-8")

        repaired_libs: List[str] = [
            f
            for f in z.namelist()
            if re.search(
                r"^(lib)?(ncarrayjit|ncarray|ncdevarray)[a-zA-Z0-9_.\-]*\.(so|dylib|dll)",
                os.path.basename(f),
            )
        ]

    if repaired_libs and pc_content:
        pc_dir: str = os.path.dirname(pc_path_in_whl)
        updated_libs: List[str] = []
        for lib in repaired_libs:
            rel_lib_path: str = os.path.relpath(lib, start=pc_dir).replace("\\", "/")
            updated_libs.append(rel_lib_path)

        # Update the includedir and libdir to point inside the wheel
        # Leave prefix as is I guess?
        whl_inc_dir: str = os.path.normpath(f"{pc_dir}/../include")
        rel_inc_dir: str = os.path.relpath(whl_inc_dir, start=pc_dir).replace("\\", "/")

        whl_lib_dir: str = os.path.normpath(os.path.dirname(repaired_libs[0]))
        rel_lib_dir: str = os.path.relpath(whl_lib_dir, start=pc_dir).replace("\\", "/")

        updated_content: str = re.sub(
            r"includedir=.*", f"includedir=${{pcfiledir}}/{rel_inc_dir}", pc_content
        )
        updated_content = re.sub(
            r"libdir=.*", f"libdir=${{pcfiledir}}/{rel_lib_dir}", updated_content
        )

        new_link_str: str = " ".join(f"${{pcfiledir}}/{rl}" for rl in updated_libs)
        updated_content = re.sub(r"Libs:.*", f"Libs: {new_link_str}", updated_content)

        temp_whl: str = whl_path + ".tmp"
        with zipfile.ZipFile(whl_path, "r") as zin:
            with zipfile.ZipFile(temp_whl, "w", compression=zin.compression) as zout:
                for item in zin.infolist():
                    if item.filename == pc_path_in_whl:
                        zout.writestr(item, updated_content.encode("utf-8"))
                    else:
                        zout.writestr(item, zin.read(item.filename))
        os.replace(temp_whl, whl_path)


def main():
    if len(sys.argv) < 3:
        print("Usage: repair_wheel.py <wheel_path> <dest_dir> [--no-exclude-core]")
        sys.exit(1)

    wheel_path: str = sys.argv[1]
    dest_dir: str = sys.argv[2]

    # Check if the exclusion flag is set. When building in the split mode you
    # do NOT exclude them, otherwise you do.
    no_exclude_core: bool = "--no-exclude-core" in sys.argv

    cmd: List[str]
    if sys.platform == "darwin":
        cmd = ["delocate-wheel"]
        if "--require-archs" in sys.argv:
            idx: int = sys.argv.index("--require-archs")
            delocate_archs: str = sys.argv[idx + 1]
            cmd.extend(["--require-archs", delocate_archs])
    else:
        cmd = [
            "auditwheel",
            "repair",
        ]

    if not no_exclude_core:
        if sys.platform == "darwin":
            cmd.extend(
                [
                    "--exclude",
                    "libncarray.1.dylib",
                    "--exclude",
                    "libncdevarray.1.dylib",
                    "--exclude",
                    "libncarrayjit.1.dylib",
                ]
            )
        else:
            cmd.extend(
                [
                    "--exclude",
                    "libncarray.so.1",
                    "--exclude",
                    "libncdevarray.so.1",
                    "--exclude",
                    "libncarrayjit.so.1",
                ]
            )
    if sys.platform == "darwin":
        cmd.extend(
            [
                "--exclude",
                "libeagereval.1.dylib",
                "-w",
                dest_dir,
                wheel_path,
            ]
        )
    else:
        cmd.extend(
            [
                "--exclude",
                "libeagereval.so.1",
                "--exclude",
                "libcuda.so.1",
                "--lib-sdir",
                ".libs",
                "-w",
                dest_dir,
                wheel_path,
            ]
        )
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    repaired_wheels: List[str] = glob.glob(os.path.join(dest_dir, "*.whl"))
    if not repaired_wheels:
        print("No repaired wheels found.")
        sys.exit(1)

    if sys.platform == "darwin":
        for whl in repaired_wheels:
            update_pc_in_repaired_wheel(whl_path=whl)
    else:
        cuda_home: str = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        builtins_src: List[str] = glob.glob(
            os.path.join(cuda_home, "lib64/libnvrtc-builtins.so*")
        )

        if not builtins_src:
            print(
                "Warning: libnvrtc-builtins.so not found in CUDA directory. Skipping injection."
            )

        for whl in repaired_wheels:
            if builtins_src:
                print(f"Injecting builtins into {whl}...")
                with zipfile.ZipFile(whl, "a") as z:
                    libs_dir: str = next(
                        (
                            os.path.dirname(name)
                            for name in z.namelist()
                            if "libnvrtc-" in os.path.basename(name)
                        ),
                        "",
                    )

                    if not libs_dir:
                        libs_dir = next(
                            (
                                os.path.dirname(name)
                                for name in z.namelist()
                                if "libcudart-" in os.path.basename(name)
                            ),
                            "ncarray.libs",
                        )

                    builtins_file: str
                    for builtins_file in builtins_src:
                        # Make sure to update symlink'd filenames appropriately
                        real_file: str = os.path.realpath(builtins_file)
                        basename: str = os.path.basename(builtins_file)
                        target_path: str = os.path.join(libs_dir, basename)
                        print(f"  Adding {real_file} -> {target_path}")
                        z.write(real_file, target_path)

            if sys.platform != "darwin":
                unmangle_wheel_libs_linux(whl_path=whl)
            update_pc_in_repaired_wheel(whl_path=whl)


if __name__ == "__main__":
    main()
