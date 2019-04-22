%% Lucas-Kanade Method Example 1
% This example uses Lucas-Kanade method on two images and calculate the
% optical flow field. 
%% Load Frames
clear all;
%load frames
fr1 = imread('D:\EIE4512\project\gtaTest1\frames\20.jpg');
fr2 = imread('D:\EIE4512\project\gtaTest1\frames\21.jpg');
fr1db = im2double(fr1);
fr2db = im2double(fr2);
rate = 1;%scale rate

figure();
subplot 211
imshow(fr1);
im1t = im2double(rgb2gray(fr1));% RGB to gray image
im1 = imresize(im1t, rate); % downsize to half

subplot 212
imshow(fr2);
im2t = im2double(rgb2gray(fr2));
im2 = imresize(im2t, rate); % downsize to half
 
%% Implementing Lucas Kanade Method
ww = 20;
w = round(ww/2);

% Lucas Kanade Here
% for each point, calculate I_x, I_y, I_t
Ix_m = conv2(im2,[-1 1; -1 1], 'valid'); % partial on x
Iy_m = conv2(im2, [-1 -1; 1 1], 'valid'); % partial on y
It_m = conv2(im1, ones(2), 'valid') + conv2(im2, -ones(2), 'valid'); % partial on t??
u_fore = zeros(size(im1));
v_fore = zeros(size(im2));
u_back = zeros(size(im1));
v_back = zeros(size(im2));
 
% within window ww * ww
for i = w+1:size(Ix_m,1)-w
   for j = w+1:size(Ix_m,2)-w
      Ix = Ix_m(i-w:i+w, j-w:j+w);
      Iy = Iy_m(i-w:i+w, j-w:j+w);
      It = It_m(i-w:i+w, j-w:j+w);
      
      Ix = Ix(:);
      Iy = Iy(:);
      b = -It(:); % get b here
    
      A = [Ix Iy]; % get A here
      nu = pinv(A)*b; % get velocity here
      
      %classify fore/back ground
      if norm(nu,2)>0.3
        u_fore(i,j)=nu(1);
        v_fore(i,j)=nu(2);
        u_back(i,j)=0;
        v_back(i,j)=0;
      else
        u_fore(i,j)=0;
        v_fore(i,j)=0;
        u_back(i,j)=nu(1);
        v_back(i,j)=nu(2);
      end      
   end
end


% stacking
frout = fr1db;

for i = 1:size(fr1,1)
    for j = 1:size(fr1,2)
        if (u_fore(i,j)~=0 || v_fore(i,j)~=0)
            frout(i,j,:) = fr1db(i,j,:)*0.5 + fr2db(i,j,:)*0.5;
        end
    end
end
figure();
imshow(frout);



% downsize u and v
u_deci = u_fore(1:10:end, 1:10:end);
v_deci = v_fore(1:10:end, 1:10:end);
% get coordinate for u and v in the original frame
[m, n] = size(im1t);
[X,Y] = meshgrid(1:n, 1:m);
X_deci = X(1:10/rate:end, 1:10/rate:end);
Y_deci = Y(1:10/rate:end, 1:10/rate:end);

%% Plot optical flow field
figure();
imshow(fr2);
hold on;
% draw the velocity vectors
quiver(X_deci, Y_deci, u_deci,v_deci, 'y')
