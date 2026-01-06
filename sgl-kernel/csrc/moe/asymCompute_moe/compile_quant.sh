# pip install pybind11
pip uninstall -y fp8_quant_ext 
rm -f /root/.local/lib/python3.12/site-packages//fp8_quant_ext.cpython-312-aarch64-linux-gnu.so
rm -rf build
rm -rf dist
rm -rf fp8_quant_ext.egg-info
python3 setup_quant.py install --user > compile.log 2>&1