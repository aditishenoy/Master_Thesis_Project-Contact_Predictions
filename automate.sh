for threshold in 2
do
	for range_mode in all short medium long
        do
                python3 unet_run_test.py 12 8 $threshold $range_mode
        done
done


