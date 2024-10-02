@echo off
setlocal enabledelayedexpansion

rem Define the precision array
set "precision_array=1e-3 1e-4"
set "noise_level=0.5"

rem Loop through each noise level
for %%n in (%noise_level%) do (
    set "noise=%%n"
    set "input_file=noisy_!noise!_downsampled_1"

    rem Loop through each precision value
    for %%p in (%precision_array%) do (
        set "precision=%%p"
        set "output_file=noise_!noise!_!precision!"

        rem Run the python script with the specified parameters
        python run.py --gpu 0 --conf confs/npull.conf --dataname !input_file!  --dir !output_file! --precision !precision!
    )
)

endlocal
