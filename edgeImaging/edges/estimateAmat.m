datafolder = '../data/testvideos/experiment_2';
fname = sprintf('%s/dark_MovieLines_green1.MOV', datafolder);
v = VideoReader(fname);
frame1 = double(readFrame(v));
frame1 = blurDnClr(frame1, 3, binomialFilter(5));
imagesc(frame1(:,:,1));
corner = ginput(1);
hold on; plot(corner(1), corner(2), 'ro');

for r = 10:5:30
    gradientAlongCircle(r, 200, frame1, corner);
end