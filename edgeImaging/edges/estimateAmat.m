addpath(genpath('rectify'));

%datafolder = '../data/testvideos/experiment_2';
datafolder = '/Users/klbouman/Downloads';

gridfile = sprintf('%s/calibrationgrid.MOV', datafolder);
do_rectify = 1;
moviefile = sprintf('%s/dark_MovieLines_green1.MOV', datafolder);

v = VideoReader(moviefile);
frame1 = double(read(v,1));

if do_rectify == 1
    v = VideoReader(gridfile);
    caliImg = read(v,100);
    [iold, jold, ii, jj, rectified_img] = rectify_image_solve(caliImg);
    %figure;imagesc(rectified_img./max(rectified_img(:)))
    frame1 = rectify_image(frame1, iold, jold, ii, jj);
end

frame1 = blurDnClr(frame1, 3, binomialFilter(5));
imagesc(frame1(:,:,1));

corner = ginput(1);
hold on; plot(corner(1), corner(2), 'ro');

vout = VideoWriter(sprintf('%s/out_green1.MOV', datafolder));
vout.FrameRate = 10;
minclip = 0;
maxclip = 1;
open(vout)

for n=1:500
    n
    
    % read the nth frame
    framen = double(read(v,n));
    if do_rectify == 1
        framen = rectify_image(framen, iold, jold, ii, jj);
    end
    framen = blurDnClr(framen, 3, binomialFilter(5));
    
    count = 1;
    for r = 10:5:30
        [rgbq(:,:,:,count), diffs(:,:,:,count)] = gradientAlongCircle(r, 200, framen, corner);
        count = count + 1;
    end
    
    %compute the average frame from all the circle differences
    outframe(1,:,:) = mean(diffs,4);
    
    %write out the video
    outframe(outframe<minclip) = minclip;
    outframe(outframe>maxclip) = maxclip;
    
    writeVideo(vout, (repmat(outframe, [100 1]) -minclip)./(maxclip-minclip));
end

close(vout);

