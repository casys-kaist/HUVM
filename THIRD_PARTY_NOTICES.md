# License scope and third-party notices

The root [MIT license](LICENSE) applies only to original contributions authored
for HUVM. It does not replace the licenses of third-party code, binaries, or
other materials. Original HUVM additions to upstream files are offered under
MIT; the upstream portions and their applicable conditions remain unchanged.
Preserve the existing copyright, permission, and disclaimer notices when
redistributing those materials.

## NVIDIA driver 460.67

`driver/` and `driver_base/` contain material from the NVIDIA Linux driver
460.67. The original package license is included verbatim in
[licenses/NVIDIA-460.67.txt](licenses/NVIDIA-460.67.txt).
Source files with their own permission grants, including MIT notices in UVM
and DRM files, retain those grants. NVIDIA material covered by the package
license is not relicensed under the root MIT license.

The package license includes a Linux/FreeBSD redistribution provision in
section 2.1.2, with a condition concerning unmodified binaries. Read the full
license for its other conditions. The `nv-kernel.o_binary` and
`nv-modeset-kernel.o_binary` files in both driver directories match the
corresponding files in the official 460.67 package byte for byte.

Original package:
[NVIDIA-Linux-x86_64-460.67.run](https://download.nvidia.com/XFree86/Linux-x86_64/460.67/NVIDIA-Linux-x86_64-460.67.run).
The included license was extracted from this package without alteration.

## CUDA sample and graphics utility code

The NVIDIA CUDA sample code in `bench/microbench/`, including `vectorAdd/`
and many `common/` helpers, carries BSD 3-Clause notices in the individual
files. Those notices remain applicable.

`bench/microbench/common/GL/` contains additional third-party material,
including GLEW, Mesa, Khronos, freeglut, GLUT, and SGI code. The files contain
their respective copyright and license notices; these are not all the same
license. In particular, `glext.h` and `glxext.h` reference SGI Free Software
License B version 1.1, while `glut.h` carries its own distribution notice.
Consult and preserve each applicable notice.

## Separately obtained dependencies

Dependencies, drivers, frameworks, and datasets obtained separately through
the setup instructions retain their respective licenses and terms. They are
not covered by HUVM's MIT grant.
