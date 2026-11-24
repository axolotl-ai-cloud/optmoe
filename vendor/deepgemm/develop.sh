# Change current directory into project root
original_dir=$(pwd)
script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

# Link CUTLASS includes
ln -sf $script_dir/third-party/cutlass/include/cutlass optmoe_deepgemm/include
ln -sf $script_dir/third-party/cutlass/include/cute optmoe_deepgemm/include

# Remove old dist file, build files, and build
rm -rf build dist
rm -rf *.egg-info
python setup.py build

# Find the .so file in build directory and create symlink in current directory
so_file=$(find build -name "*.so" -type f | head -n 1)
if [ -n "$so_file" ]; then
    ln -sf "../$so_file" optmoe_deepgemm/
else
    echo "Error: No SO file found in build directory" >&2
    exit 1
fi

# Open users' original directory
cd "$original_dir"
