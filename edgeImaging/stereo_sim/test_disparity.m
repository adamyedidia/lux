close all;

baseline = 1;
depth = 3;
width = 5;
x = linspace(-width/2,width/2,100);
yflat = linspace(depth,depth,100);
ydiag = linspace(depth,depth*3,100);
  
c1 = [-baseline/2, 0];
c2 = [baseline/2, 0];
asig = pi/100;
quantize = 8;

%%
% a1 is always less than a2

a1_flat0 = atan2((yflat-c1(2)), (x-c1(1)));
% a1_flat = a1_flat0 + randn(size(x))*asig;
a1_flat = round(quantize*a1_flat0)/quantize;
a2_flat0 = atan2((yflat-c2(2)), (x-c2(1))); 
% a2_flat0 = a1_flat0 + mean(a2_flat0(:)) - mean(a1_flat0(:)); % constant angle offset
% a2_flat = a2_flat0 + randn(size(x))*asig;
a2_flat = round(quantize*a2_flat0)/quantize;

depths_flat = 1 ./ (cot(a1_flat) + cot(pi - a2_flat));
xlocs_flat = depths_flat .* cot(a1_flat) + c1(1);
error_flat = asig .* depths_flat.^2 .* sqrt(csc(a1_flat).^4 + csc(pi - a2_flat).^4);

figure;
% plot(x,a1, 'b-'); hold on; plot(x,a2, 'r-'); 
subplot(211);
plot(a1_flat/pi, a2_flat/pi, 'o'); hold on;
plot(a1_flat0/pi, a2_flat0/pi);
plot(a1_flat0/pi, a1_flat0/pi, '.');
xlabel('\alpha_1/\pi');
ylabel('\alpha_2/\pi');
legend('noisy angles', 'ground truth angles', '\alpha_1 = \alpha_2', 'Location', 'southeast');

subplot(212);
errorbar(xlocs_flat, depths_flat, error_flat, 'o'); hold on;
plot(x, yflat);
xlabel('x');
ylabel('depth');
legend('estimates', 'ground truth', 'Location', 'southeast');

%%

a1_diag0 = atan2((ydiag-c1(2)), (x-c1(1)));
% a1_diag = a1_diag0 + randn(size(x))*asig;
a1_diag = round(quantize*a1_diag0)/quantize;
a2_diag0 = atan2((ydiag-c2(2)), (x-c2(1)));
% a2_diag = a2_diag0 + randn(size(x))*asig;
a2_diag = round(quantize*a2_diag0)/quantize;


depths_diag = 1 ./ (cot(a1_diag) + cot(pi - a2_diag));
xlocs_diag = depths_diag .* cot(a1_diag) + c1(1);
error_diag = asig .* depths_diag.^2 .* sqrt(csc(a1_diag).^4 + csc(pi - a2_diag).^4);


figure;
subplot(211);
plot(a1_diag/pi, a2_diag/pi, 'o'); hold on;
plot(a1_diag0/pi, a2_diag0/pi);
plot(a1_diag0/pi, a1_diag0/pi, '.');
xlabel('\alpha_1/\pi');
ylabel('\alpha_2/\pi');
legend('noisy angles', 'ground truth angles', '\alpha_1 = \alpha_2', 'Location', 'southeast');

subplot(212);
errorbar(xlocs_diag, depths_diag, error_diag, 'o'); hold on;
plot(x, ydiag);
xlabel('x');
ylabel('depth');
legend('estimates', 'ground truth', 'Location', 'southeast');
