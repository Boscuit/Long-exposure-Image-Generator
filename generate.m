im1 = imread('D:\EIE4512\project\circletest\(1).png');
N=6;
frlist=zeros(size(im1,1),size(im1,2),3,N);
frgraylist=zeros(size(im1,1),size(im1,2),N);
frout = im2double(im1);
fgof = zeros(size(im1,1),size(im1,2),2);%accumulated foreground optical flow
fgoflist = zeros(size(im1,1),size(im1,2),2,N);
rate=1;
threshold=0.3;

frlist_b = zeros(size(frlist));
frgraylist_b = zeros(size(frgraylist));


for i = 1:N
    name = strcat('D:\EIE4512\project\circletest\(',num2str(i),').png');
    im = imread(name);
    frlist(:,:,:,i) = im2double(im);
    frgraylist(:,:,i) = rgb2gray(frlist(:,:,:,i));
end


frlist_b(:,:,:,N) = frlist(:,:,:,N);
frgraylist_b(:,:,N) = frgraylist(:,:,N);

for p = 1:N-1
    im1 = frgraylist(:,:,p);
    im2 = frgraylist(:,:,p+1);
    im1 = imresize(im1, rate); % rescale
    im2 = imresize(im2, rate); % rescale
    opticalflow = getopticalflow2(im1,im2,threshold);
    fgoflist(:,:,:,p) = opticalflow(:,:,1:2);
    fgof = fgof + fgoflist(:,:,:,p);

    % stacking(1-2)
%     frout = stack1(frout,frlist(:,:,:,p+1),fgoflist(:,:,1,p),fgoflist(:,:,2,p));
%     frout = stack2(frout,frlist(:,:,:,p+1),fgoflist(:,:,1,p),fgoflist(:,:,2,p),threshold);
    U = opticalflow(:,:,1);
    V = opticalflow(:,:,2);
    mean_u = mean(U(:));
    mean_v = mean(V(:));
    sita = atan(mean_v/mean_u)*(180/pi);
    norm = sqrt(mean_u^2+mean_v^2);
    H = fspecial('motion',norm*10,sita);
    frlist_b(:,:,:,p) = imfilter(frlist(:,:,:,p),H,'replicate');
    frgraylist_b(:,:,p) = imfilter(frgraylist(:,:,p),H,'replicate');
    %stack1
    frout = stack1(frout,frlist_b(:,:,:,p),fgoflist(:,:,1,p),fgoflist(:,:,2,p));
%     figure;
%     imshow(frout);
    
end
    
%stack(3-5)
% frout = stack6(frout,frgraylist_b,frlist_b,fgof); 



figure();
imshow(frout);

% downsize u and v
u_deci = fgof(1:10:end, 1:10:end, 1);
v_deci = fgof(1:10:end, 1:10:end, 2);
% get coordinate for u and v in the original frame
[m, n] = size(im1);
[X,Y] = meshgrid(1:n, 1:m);
X_deci = X(1:10/rate:end, 1:10/rate:end);
Y_deci = Y(1:10/rate:end, 1:10/rate:end);
% Plot optical flow field
figure();
imshow(frlist(:,:,:,1));
hold on;
% draw the velocity vectors
quiver(X_deci, Y_deci, u_deci,v_deci, 'b')



%---------get optical flow pixel by pixel---------
function [opticalflow] = getopticalflow1(im1,im2,threshold)
    opticalflow = zeros([size(im1),4]);
    ww = 20;
    w = round(ww/2);

    % Lucas Kanade Here
    % for each point, calculate I_x, I_y, I_t
    Ix_m = conv2(im2, [-1 1; -1 1], 'valid'); % partial on x
    Iy_m = conv2(im2, [-1 -1; 1 1], 'valid'); % partial on y
    It_m = conv2(im1, ones(2), 'valid') + conv2(im2, -ones(2), 'valid'); % partial on t??
    u_fore = zeros(size(im1));
    v_fore = zeros(size(im1));
    u_back = zeros(size(im1));
    v_back = zeros(size(im1));

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
          if norm(nu,2)>threshold
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
    opticalflow(:,:,1) = u_fore;
    opticalflow(:,:,2) = v_fore;
    opticalflow(:,:,3) = u_back;
    opticalflow(:,:,4) = v_back;
end


%---------get optical flow with superpixel segementation---------
function [opticalflow] = getopticalflow2(im1,im2,threshold)
    opticalflow = zeros([size(im1),4]);
    ww = 20;
    w = round(ww/2);
    %get superpixel and pad in boudrary with 0
    [L,N] = superpixels(im1,500);% L superpixel-map with label 
%     figure
%     BW = boundarymask(L);
%     imshow(imoverlay(im1,BW,'cyan'),'InitialMagnification',67)
    
    Lpad = zeros(size(L));
    Lpad(w+1:size(im1,1)-w-1,w+1:size(im1,2)-w-1) = L(w+1:size(im1,1)-w-1,w+1:size(im1,2)-w-1);
    %get all location with each label
    IDXpad = label2idx(Lpad);%for calculation
    IDX = label2idx(L);%for assignment

    % Lucas Kanade Here
    % for each point, calculate I_x, I_y, I_t
    Ix_m = conv2(im2, [-1 1; -1 1], 'valid'); % partial on x
    Iy_m = conv2(im2, [-1 -1; 1 1], 'valid'); % partial on y
    It_m = conv2(im1, ones(2), 'valid') + conv2(im2, -ones(2), 'valid'); % partial on t??
    u_fore = zeros(size(im1));
    v_fore = zeros(size(im1));
    u_back = zeros(size(im1));
    v_back = zeros(size(im1));

    % within window ww * ww
    for labelVal = 1:N
        allidxpad = IDXpad{labelVal};
        allidx = IDX{labelVal};
        %within the region which pixel is selected? {round(size(allidxpad,1)/2)}--middle one
%         idx = allidxpad(1); %--first one
        idx = allidxpad(round(size(allidxpad,1)/2)); %--middle one
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
        if norm(nu,2)>threshold
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
    opticalflow(:,:,1) = u_fore;
    opticalflow(:,:,2) = v_fore;
    opticalflow(:,:,3) = u_back;
    opticalflow(:,:,4) = v_back;
end


%---------stack method mean filter(pixel by pixel)---------
function[frout] = stack1(base,top,of_u,of_v)
    frout = base;
%     figure;
%     subplot(211);
%     imshow(top);
%     subplot(212);
%     imshow(base);
    for i = 1:size(base,1)
        for j = 1:size(base,2)
            if (of_u(i,j)~=0 || of_v(i,j)~=0)
                frout(i,j,:) = (base(i,j,:) + top(i,j,:))/2;
            end
        end
    end
end


%---------stack method mean filter(matix operation)---------
function[frout] = stack2(base,top,of_u,of_v,threshold)
    fgofNlist = zeros(size(base,1),size(base,2));
    fgofNlist(:,:) = sqrt(of_u.^2 + of_v.^2);%get norm
    fgofNlist(:,:) = fgofNlist(:,:)/threshold;%normalization (0 or >=1)
    fgofNlist(:,:) = fgofNlist(:,:)./(fgofNlist(:,:)+1);%normalization (0 or [0.5,1))
    frout = base.*(1-fgofNlist(:,:)) +  top.*fgofNlist(:,:);
end


%---------stack method median filter(pixel by pixel)---------
function[frout] = stack3(base,frgraylist,frlist,fgof)
    frNum = zeros(size(fgof,1),size(fgof,2));%selection record
    frout = base;
    for i = 1:size(fgof,1)
        for j = 1:size(fgof,2)
            if (fgof(i,j,1)~=0 || fgof(i,j,2)~=0)
                label = find(frgraylist(i,j,:)==median(frgraylist(i,j,:)));
                num = label(1);%the first frame have median graylevel 
                frNum(i,j) = num;%record
                frout(i,j,:) = frlist(i,j,:,num);%assignment
            end
        end
    end
end

%---------stack method min filter(pixel by pixel)---------
function[frout] = stack4(base,frgraylist,frlist,fgof)
    frNum = zeros(size(fgof,1),size(fgof,2));%selection record
    frout = base;
    for i = 1:size(fgof,1)
        for j = 1:size(fgof,2)
            if (fgof(i,j,1)~=0 || fgof(i,j,2)~=0)
                label = find(frgraylist(i,j,:)==min(frgraylist(i,j,:)));
                num = label(1);%the first frame have minimum graylevel 
                frNum(i,j) = num;%record
                frout(i,j,:) = frlist(i,j,:,num);%assignment
            end
        end
    end
end

%---------stack method max filter(pixel by pixel)---------
function[frout] = stack5(base,frgraylist,frlist,fgof)
    frNum = zeros(size(fgof,1),size(fgof,2));%selection record
    frout = base;
    for i = 1:size(fgof,1)
        for j = 1:size(fgof,2)
            if (fgof(i,j,1)~=0 || fgof(i,j,2)~=0)
                label = find(frgraylist(i,j,:)==max(frgraylist(i,j,:)));
                num = label(1);%the first frame have maximum graylevel 
                frNum(i,j) = num;%record
                frout(i,j,:) = frlist(i,j,:,num);%assignment
            end
        end
    end
end


%---------generate method(pixel by pixel)---------
function[frout] = stack6(base,frgraylist,frlist,fgof)
    frNum = zeros(size(fgof,1),size(fgof,2));%selection record
    frout = base;
    for i = 1:size(fgof,1)
        for j = 1:size(fgof,2)
            if (fgof(i,j,1)~=0 || fgof(i,j,2)~=0)
                label = find(frgraylist(i,j,:)==min(frgraylist(i,j,:)));
                num = label(1);%the first frame have minimum graylevel 
                frNum(i,j) = num;%record
                frout(i,j,:) = frlist(i,j,:,num);%assignment
            end
        end
    end
end