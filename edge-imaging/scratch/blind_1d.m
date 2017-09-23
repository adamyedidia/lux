close all;

n = 200;

signal = zeros([200, 1]);
signal(10) = 1;
occluder = zeros([200, 1]);
occluder(end/2 - 5:end/2 + 5) = 1;

obs = conv(signal, occluder, 'same');
subplot(311); plot(signal, 'o');
subplot(312); plot(occluder, 'o');
subplot(313); plot(obs, 'o');