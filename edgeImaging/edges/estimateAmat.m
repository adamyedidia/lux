addpath(genpath('rectify'));
    
datafolder = '../data/testvideos/experiment_2';
gridfile = sprintf('%s/calibrationgrid.MOV', datafolder);
do_rectify = 0;
moviefile = sprintf('%s/dark_MovieLines_green1.MOV', datafolder);
 
v = VideoReader(moviefile);
frame1 = double(readFrame(v));

if do_rectify == 1
    v = VideoReader(gridfile);
    caliImg = readframe(v);
    [iold, jold, ii, jj, rectified_img] = rectify_image_solve(caliImg); 
    figure; imagesc(rectified_img./max(rectified_img(:)))
    frame1 = rectify_image(frame1, iold, jold, ii, jj);
end

frame1 = blurDnClr(frame1, 3, binomialFilter(5));
imagesc(frame1(:,:,1));

corner = ginput(1);
hold on; plot(corner(1), corner(2), 'ro');

rs = 10:5:10;
nsamples = 200
all_diffs = zeros([nsamples-1, 3, length(rs)]);

for i = 1:length(rs)
    [~, diffs] = gradientAlongCircle(rs(i), nsamples, frame1, corner);
    all_diffs(:,:,i) = diffs;
end