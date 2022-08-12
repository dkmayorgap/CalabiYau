echo "Computing chi using gFS:"
python Quintic_compute_Euler.py --path_to_points QuinticData --path_to_riems riemvaluesfs --mode val
echo "------------------------------------"


echo "Computing chi using gpred:"
python Quintic_compute_Euler.py --path_to_points QuinticData --path_to_riems riemvalues --mode val
echo "------------------------------------"


echo "Done!"
