# pip install pybind11
pip uninstall -y AsymCompute 
rm -f /root/.local/lib/python3.12/site-packages/AsymCompute.cpython-312-aarch64-linux-gnu.so
rm -rf build
rm -rf dist
rm -rf AsymCompute.egg-info
python3 setup.py install --user > compile.log 2>&1