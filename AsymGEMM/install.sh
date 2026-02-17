set -euo pipefail

# Change current directory into project root
original_dir=$(pwd)
script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

# Remove old dist file, build files, and install
rm -rf build dist
rm -rf *.egg-info
# Remove stale in-tree extension modules so local import from repo root won't pick old binaries
rm -f asym_gemm/_C*.so deep_gemm/_C*.so

# Build local extension in-place (for importing from this repo), then build wheel for site-packages
python setup.py build_ext --inplace
python setup.py bdist_wheel
pip install dist/*.whl --force-reinstall

# Open users' original directory
cd "$original_dir"
