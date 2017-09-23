function writeReconVideo(outfile, outframes)
vout = VideoWriter(outfile);
vout.FrameRate = 10;
open(vout);
for i = 1:nout
    outi = outframes(i,:,:);
    outh = round(size(outi,2)/2);
    writeVideo(vout, repmat(outi, [outh, 1]));
end
close(vout);
end