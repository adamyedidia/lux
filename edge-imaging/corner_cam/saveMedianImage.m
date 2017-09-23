function outfile = saveMedianImage(overwrite, isvid, moviefile, start, endframe, step)

if nargin < 6
    step = 1;
end

switch isvid
case 1
    split = strsplit(moviefile, '.');
    outfile = char(strcat(split(1), '_median_img.mat'));
    vsrc = VideoReader(moviefile);
    framesize = size(read(vsrc, 1));
case 0
    outfile = fullfile(moviefile, 'median_img.mat');
    vsrc = moviefile;
    framesize = size(imread(fullfile(vsrc, 'photo_1_2.CR2')));
case -1
    outfile = sprintf('%s_median_img.mat', moviefile);
    vsrc = moviefile;
    fname = sprintf('%s-%4.4d.pgm', vsrc, 1);
    framesize = size(imread(fname));
end

if ~overwrite && exist(outfile, 'file')
    return
end

fprintf('Saving median image in %s\n', outfile);

count = 0;
colorhist = zeros([framesize(1), framesize(2), framesize(3), 256]);
[ii, jj, kk] = ndgrid(1:framesize(1),1:framesize(2),1:framesize(3));
for n = start:step:endframe
    frame = getFrame(isvid, vsrc, n, 1, framesize);
    % colorvalues are 0 to 255, index must be 1 to 256
    idx = sub2ind(size(colorhist), ii, jj, kk, frame+1);
    colorhist(idx) = colorhist(idx) + 1;
    count = count + 1;
end

cumhist = cumsum(colorhist, 4);
isless = cumhist < count/2;
med_img = sum(isless, 4) + 1;
save(outfile, 'med_img');
end
