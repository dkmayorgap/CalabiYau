
echo "Running point generation..."
python Quintic_pointgen.py --output_path QuinticData --num_pts 500000 --val_split 0.2

echo "Running ML training..."
RET=1
until [ ${RET} -eq 0 ]; do
    python Quintic_train.py --path_to_points QuinticData --output_directory QuinticModel
    RET=$?
    sleep 10
done


# TODO: watch return cond before end
echo "------------------------------------"
echo "Computing Riemann tensors for gFS..."
mkdir riemvaluesfs

RET=1
until [ ${RET} -eq 0 ]; do
    python Quintic_calc_riem.py --path_to_points QuinticData --path_to_model QuinticModel --path_to_output riemvaluesfs --metric fs --mode val
    RET=$?
    sleep 10
done

echo "Computing chi using gFS:"
python Quintic_compute_Euler.py --path_to_points QuinticData --path_to_riems riemvaluesfs --mode val
echo "------------------------------------"

echo "Computing Riemann tensors for gpred..."
mkdir riemvalues

RET=1
until [ ${RET} -eq 0 ]; do
    python Quintic_calc_riem.py --path_to_points QuinticData --path_to_model QuinticModel --path_to_output riemvalues --metric pred --mode val
    RET=$?
    sleep 10
done

echo "Computing chi using gpred:"
python3 Quintic_compute_Euler.py --path_to_points QuinticData --path_to_riems riemvalues --mode val
echo "------------------------------------"


echo "Done!"
