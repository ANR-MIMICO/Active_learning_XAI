cd scripts
conda run --no-capture-output -n clean-sego python al_tp_v1.py
conda run --no-capture-output -n clean-sego python al_tp_v2.py
conda run --no-capture-output -n clean-sego python al_tp_v2-2dim.py
conda run --no-capture-output -n clean-sego python al_tp_v3.py
cd ..
echo "All Active Learning scripts finished executing."
