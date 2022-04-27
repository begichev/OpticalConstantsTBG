set pm3d map
set xrange [10:19]
set xlabel "Angle (deg.)"
set yrange [1300:2000]
set ylabel "Energy (meV)"
splot 'sigmacopy.dat' u 1:2:4
