set terminal pdf enhanced
set output output_file

set xrange[minx:maxx]
set yrange[miny:maxy]
set title title_name
set view map
set key top outside

plot "output0.csv" with points notitle ps psvalue pt 7 lc rgb "yellow",\
"output1.csv" with points notitle ps psvalue pt 7 lc rgb "orange",\
"output2.csv" with points notitle ps psvalue pt 7 lc rgb "red",\
"output3.csv" with points notitle ps psvalue pt 7 lc rgb "magenta",\
"output4.csv" with points notitle ps psvalue pt 7 lc rgb "purple",\
"output5.csv" with points notitle ps psvalue pt 7 lc rgb "dark-violet",\
"output6.csv" with points notitle ps psvalue pt 7 lc rgb "blue",\
"output7.csv" with points notitle ps psvalue pt 7 lc rgb "cyan",\
"output8.csv" with points notitle ps psvalue pt 7 lc rgb "dark-green",\
"output9.csv" with points notitle ps psvalue pt 7 lc rgb "green",\
1/0 with points title "0" ps psvalue2 pt 7 lc rgb "yellow",\
1/0 with points title "1" ps psvalue2 pt 7 lc rgb "orange",\
1/0 with points title "2" ps psvalue2 pt 7 lc rgb "red",\
1/0 with points title "3" ps psvalue2 pt 7 lc rgb "magenta",\
1/0 with points title "4" ps psvalue2 pt 7 lc rgb "purple",\
1/0 with points title "5" ps psvalue2 pt 7 lc rgb "dark-violet",\
1/0 with points title "6" ps psvalue2 pt 7 lc rgb "blue",\
1/0 with points title "7" ps psvalue2 pt 7 lc rgb "cyan",\
1/0 with points title "8" ps psvalue2 pt 7 lc rgb "dark-green",\
1/0 with points title "9" ps psvalue2 pt 7 lc rgb "green"
