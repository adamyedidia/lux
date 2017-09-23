function meanimg = getMeanFrame(vsrc, downlevs, fps, start_t, end_t)
vsrc = VideoReader(vsrc);
times = start_t: 1/fps :end_t;

vsrc.CurrentTime = times(1);
filt = binomialFilter(5);
meanimg = blurDnClr(double(readFrame(vsrc)), downlevs, filt);
for i = 2:length(times)
    vsrc.CurrentTime = times(i);
    meanimg = meanimg + blurDnClr(double(readFrame(vsrc)), downlevs, filt);
end
meanimg = meanimg / length(times);
end