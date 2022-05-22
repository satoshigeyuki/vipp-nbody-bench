set terminal png
set output 'output.png'
set datafile separator ","

set pm3d
set pm3d map
set ticslevel 0
set yrange[*:*] reverse
set cbrange[min_value:max_value]
set palette defined (min_value "white", 0.05 "red", max_value "violet")
splot "output.csv" with pm3d
