
% define the number of pixels you are observing in each direction on the
% ground and on the ceiling
npx = 50;
ceilnpx = 50; 

% load in a sample ground picture
img = imresize(double(imread('circles.png')),[npx npx]);
figure;imagesc(img);title('original image'); 

% define the amount of noise in the image and the amount of regularizer you
% add in
sigma = 1; 
beta = 10; 

% true if you want to see the integral mask images
view_integral_mask = true; 

%height of the ceiling
ceil_height = 1; 

% where you are looking on the ceiling
x_ceil_locs = linspace(0,1,ceilnpx); 
y_ceil_locs = linspace(0,1,ceilnpx); 

% the top left and right corners of the door (x, y, z)
door_corner1 = [1.1,0,.75]; 
door_corner2 = [2.1,0,.75]; 

%where the image is on the floor
[query_y, query_x] = ndgrid(linspace(-3,1,npx), linspace(0,8,npx));

%%

% get the x and y point of each ceiling location
[yy, xx] = ndgrid(y_ceil_locs, x_ceil_locs); 

% getting the intersection of the first corner point
xvec_1 = door_corner1(1) - xx;
yvec_1 = door_corner1(2) - yy; 
zvec_1 = door_corner1(3) - ceil_height; 
% figure out where the line from the ceiling to the door corner intersects
% with the ground
alpha_1 = -ceil_height/zvec_1;
x_inter_1 = xx + alpha_1*xvec_1; 
y_inter_1 = yy + alpha_1*yvec_1; 

% getting the intersection of the second corner point
xvec_2 = door_corner2(1) - xx;
yvec_2 = door_corner2(2) - yy; 
zvec_2 = door_corner2(3) - ceil_height; 
%figure out where the line from the ceiling to the door corner intersects
% with the ground
alpha_2 = -ceil_height/zvec_2;
x_inter_2 = xx + alpha_2*xvec_2; 
y_inter_2 = yy + alpha_2*yvec_2; 



%%plot those intersection points
%figure;plot(x_inter_1(:), y_inter_1(:), 'ro');
%hold on;plot(x_inter_2(:), y_inter_2(:), '+');


%%plot the lines and door corners for a single point
%figure;
%plot3([xx(1,1) x_inter_1(1,1)],[yy(1,1) y_inter_1(1,1)], [ceil_height 0])
%hold on; plot3([xx(1,1) x_inter_2(1,1)],[yy(1,1) y_inter_2(1,1)], [ceil_height 0])
%hold on; plot3(door_corner1(1), door_corner1(2), door_corner1(3), 'ko'); 
%hold on; plot3(door_corner2(1), door_corner2(2), door_corner2(3), 'ko'); 

%%

% calculate the transfer matrix A 
A = zeros(numel(x_inter_1), numel(query_y)); 
count = 1; 
for i =1:size(x_inter_1,1);
    i
    for j =1:size(x_inter_1,2);
        xv = [x_inter_1(i,j); x_inter_2(i,j); door_corner2(1); door_corner1(1)];
        yv = [y_inter_1(i,j); y_inter_2(i,j); door_corner2(2); door_corner1(2)];
        in = inpolygon(query_x,query_y,xv,yv);
        
        if view_integral_mask
            imagesc(in); title('Integral Mask'); pause(0.01); 
        end
        
        A(count,:) = double(in(:)); 
        count = count + 1; 
    end
end

%%

% get the observations
obs = A*img(:);
obs_noise = obs + sigma*randn(size(obs)); 
figure;imagesc(reshape(obs_noise, [ceilnpx ceilnpx])); title('The ceiling image'); 

% get a matrix that computes the gradient of the image to regularize
for r=1:npx
    P_row{r} = double(diag(ones(npx,1)) - diag(ones(npx-1,1),1)); 
end
P = blkdiag(P_row{:}); 
figure;imagesc(reshape(P*img(:), [npx npx])); title('P matrix extracts vertical derivative'); 

% try to recover the image
recoveredImg_2 = ((1/sigma^2)*A'*A + beta*P'*P)\((1/sigma^2)*A'*obs_noise(:));
figure;imagesc(reshape(recoveredImg_2, [npx npx])); title('Recovered floor image with just a vertical derivative regularizer');



