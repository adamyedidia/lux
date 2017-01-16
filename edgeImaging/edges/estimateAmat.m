datafolder = '../data/testvideos/experiment_2';
fname = sprintf('%s/dark_IMG_3294.MOV', datafolder);
v = VideoReader(fname);
frame1 = double(readFrame(v));
frame1 = blurDnClr(blurDnClr(blurDnClr(frame1)));
imagesc(frame1(:,:,1));
[corner_x, corner_y] = ginput(1);

r = 10;
nsamples = 200;
angles = linspace(0, pi/2, nsamples);
xq = corner_x + r * cos(angles);
yq = corner_y + r * sin(angles);
hold on;
plot(xq, yq, 'r');

[yy, xx] = ndgrid(1:size(frame1, 1), 1:size(frame1, 2));
rgbq = zeros([size(angles, 2), size(frame1, 3)]);
for i = 1:size(rgbq,2)
    rgbq(:,i) = interp2(xx, yy, frame1(:,:,i), xq, yq);
end

