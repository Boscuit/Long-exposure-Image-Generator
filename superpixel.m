rate = 1;
ww = 20;
w = round(ww/2);

fr1 = imread('D:\EIE4512\project\gtaTest1\frames\20.jpg');
fr2 = imread('D:\EIE4512\project\gtaTest1\frames\21.jpg');
im1 = im2double(rgb2gray(fr1));
im2 = im2double(rgb2gray(fr2));

[L,N] = superpixels(im1,200);
figure
BW = boundarymask(L);
imshow(imoverlay(fr1,BW,'cyan'),'InitialMagnification',67)

Lpad = zeros(size(L));
Lpad(w+1:size(fr1,1)-w-1,w+1:size(fr1,2)-w-1) = L(w+1:size(fr1,1)-w-1,w+1:size(fr1,2)-w-1);

IDX = label2idx(Lpad);

% Lucas Kanade Here
% for each point, calculate I_x, I_y, I_t
Ix_m = conv2(im2, [-1 1; -1 1], 'valid'); % partial on x
Iy_m = conv2(im2, [-1 -1; 1 1], 'valid'); % partial on y
It_m = conv2(im1, ones(2), 'valid') + conv2(im2, -ones(2), 'valid'); % partial on t??
u_fore = zeros(size(im1));
v_fore = zeros(size(im1));
u_back = zeros(size(im1));
v_back = zeros(size(im1));
    
    
for labelVal = 1:N
    allidx = IDX{labelVal};
    idx = allidx(1); %within the region which pixel is selected? {round(size(allidx,1)/2)}--median
    j = ceil(idx/size(L,1));
    i = idx - (j-1)*size(L,1);
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
        u_fore(allidx)=nu(1);
        v_fore(allidx)=nu(2);
        u_back(allidx)=0;
        v_back(allidx)=0;
    else
        u_fore(allidx)=0;
        v_fore(allidx)=0;
        u_back(allidx)=nu(1);
        v_back(allidx)=nu(2);
    end  
end

% downsize u and v
u_deci = u_fore(1:10:end, 1:10:end);
v_deci = v_fore(1:10:end, 1:10:end);
% get coordinate for u and v in the original frame
[m, n] = size(im1);
[X,Y] = meshgrid(1:n, 1:m);
X_deci = X(1:10/rate:end, 1:10/rate:end);
Y_deci = Y(1:10/rate:end, 1:10/rate:end);

% Plot optical flow field
figure();
imshow(fr1);
hold on;
% draw the velocity vectors
quiver(X_deci, Y_deci, u_deci,v_deci, 'y')