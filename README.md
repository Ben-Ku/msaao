Implementation of multi-resolution screen space ambient occlusion algorithm 
sans the temporal filtering component.

Paper describing method: https://sci.utah.edu/~duong/papers/hoang-mssao-tvc-2012-paper.pdf.

Should run basically out of the box with rust installed just run "cargo run
--release" i command line from repos top folder

Controls:
move cam - wasd qe
rotate cam - ijkl
switch scene - arrows
load saved cam - y
save current cam - z
reset cam - r
select ao resolution - 1 to 5 
