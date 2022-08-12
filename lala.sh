RET=1
until [ ${RET} -eq 0 ]; do
    python Quintic_calc_riem.py --path_to_points QuinticData --path_to_model QuinticModel --path_to_output riemvaluesfs --batch_size 250 --metric fs --mode val
    RET=$?
    sleep 10
done