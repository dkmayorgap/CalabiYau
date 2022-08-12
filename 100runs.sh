RET=5
until [ ${RET} -eq 0 ]; do
    python newriemann.py
    RET=RET-1
    sleep 10
done