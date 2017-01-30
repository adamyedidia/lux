function processExperimentVideos(moviefile, gridfile, backfile)
v = VideoReader(moviefile);
endframe = v.NumberOfFrames;
if nargin < 3
    % use the mean of the moviefile for the background
    background = zeros(size(frame1));
    count = 0;
    for n = 5*60:endframe
        background = background + double(read(v,n));
        count = count + 1;
    end
    background = background / count;
else
    vback = VideoReader(backfile);
    background = double(read(vback, floor(vback.NumberOfFrames/2)));
end
mean_pixel = mean(mean(background, 1), 2);

% solve for rectification
vcali = VideoReader(gridfile);
caliImg = read(vcali,100);
[iold, jold, ii, jj, ~] = rectify_image_solve(caliImg);
frame1 = rectify_image(frame1, iold, jold, ii, jj);
background = rectify_image(background, iold, jold, ii, jj);

figure; imagesc(frame1(:,:,1));
corner = ginput(1);
hold on; plot(corner(1), corner(2), 'ro');

clear('v', 'vback', 'vcali'); % clear video readers
outfile = moviefile(1:end-4); % same name, without .MOV extension
save(outfile);
end