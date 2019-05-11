excomprate = 0.5;
ofcomprate = 1;%scale rate
threshold = 3;

obj = VideoReader('D://EIE4512//project//realTest//test (8).mp4');
frselect = [120,90];

frlist = zeros(obj.Height*excomprate,obj.Width*excomprate,3,frselect(2));
frgraylist = zeros(obj.Height*excomprate,obj.Width*excomprate,frselect(2));


for k = 1:frselect(2)
    frame = read(obj,k+frselect(1)-1);
    frame = imresize(frame,excomprate);
    frlist(:,:,:,k) = im2double(frame);
    frgraylist(:,:,k) = rgb2gray(frlist(:,:,:,k));
    %imwrite(frame,strcat('D:\EIE4512\project\gtaTest1\frames\',num2str(k),'.jpg'),'jpg');
end

for p = 1:30
    frselect = [120+3*(p-1),3]; %frnumber[framestart,number]
    disp(['runing small group ',num2str(p),'.'])
    frout= longexpsure(frlist(:,:,:,3*p-2:3*p),frgraylist(:,:,3*p-2:3*p),ofcomprate,threshold);
    froutlist(:,:,:,p) = frout;
    froutgraylist(:,:,p) = rgb2gray(frout);
%     figure();
%     imshow(froutlist(:,:,:,p));  
end

for p = 1:10
    disp(['runing big group ',num2str(p),'.'])
    froutlist2(:,:,:,p) = longexpsure(froutlist(:,:,:,3*p-2:3*p),froutgraylist(:,:,3*p-2:3*p),ofcomprate,threshold);
    figure();
    imshow(froutlist2(:,:,:,p));  
end

function [frout] = longexpsure(frlist,frgraylist,ofcomprate,threshold)
frlist_b = zeros(size(frlist));
frgraylist_b = zeros(size(frgraylist));
fgof = zeros(size(frlist,1),size(frlist,2),2);%accumulated foreground optical flow
fgoflist = zeros(size(frlist,1),size(frlist,2),2,size(frlist,4));
frout = frlist(:,:,:,1);
frlist_b(:,:,:,size(frlist,4)) = frlist(:,:,:,size(frlist,4));
frgraylist_b(:,:,size(frlist,4)) = frgraylist(:,:,size(frlist,4));
    
for p = 1:size(frlist,4)-1
    im1 = frgraylist(:,:,p);
    im2 = frgraylist(:,:,p+1);
    im1 = imresize(im1, ofcomprate); % rescale
    im2 = imresize(im2, ofcomprate); % rescale
    [opticalflow,IDX] = getopticalflow2(im1,im2,threshold);
    disp(['runing frame ',num2str(p),'.'])
    fgoflist(:,:,:,p) = opticalflow(:,:,1:2);
    fgof = fgof + fgoflist(:,:,:,p);

    % stacking(1-2)
%     frout = stack1(frout,frlist(:,:,:,p+1),fgoflist(:,:,1,p),fgoflist(:,:,2,p),p);
%     frout = stack2(frout,frlist(:,:,:,p+1),fgoflist(:,:,1,p),fgoflist(:,:,2,p),threshold);
%     figure();
%     imshow(frout);

    %stack1
%     frout = stack1(frout,frlist(:,:,:,p),fgoflist(:,:,1,p),fgoflist(:,:,2,p),p);

end
    
%stack(3-5)
  frout = stack5(frout,frgraylist,frlist,fgof);
% frout = getmotionblur2(frout,IDX,fgof(:,:,1),fgof(:,:,2));
% figure();
% imshow(frout);

% % downsize u and v
% u_deci = fgof(1:10:end, 1:10:end, 1);
% v_deci = fgof(1:10:end, 1:10:end, 2);
% % get coordinate for u and v in the original frame
% [m, n] = size(im1);
% [X,Y] = meshgrid(1:n, 1:m);
% X_deci = X(1:10/ofcomprate:end, 1:10/ofcomprate:end);
% Y_deci = Y(1:10/ofcomprate:end, 1:10/ofcomprate:end);
% % Plot optical flow field
% figure();
% imshow(frlist(:,:,:,1));
% hold on;
% % draw the velocity vectors
% quiver(X_deci, Y_deci, u_deci,v_deci, 'y')

end

%---------get optical flow with superpixel segementation---------
function [opticalflow,IDX] = getopticalflow2(im1,im2,threshold)
    opticalflow = zeros([size(im1),4]);
    ww = 10;
    w = round(ww/2);
    %get superpixel and pad in boudrary with 0
    [L,N] = superpixels(im1,400);% L superpixel-map with label 
%     figure
%     BW = boundarymask(L);
%     imshow(imoverlay(im1,BW,'cyan'),'InitialMagnification',67)

    Lpad = zeros(size(L));
    Lpad(w+1:size(im1,1)-w-1,w+1:size(im1,2)-w-1) = L(w+1:size(im1,1)-w-1,w+1:size(im1,2)-w-1);%pad in boudary with 0
    
    %get those pixels with the same label
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
        if size(allidxpad,1)~=0
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
    end
    opticalflow(:,:,1) = u_fore;
    opticalflow(:,:,2) = v_fore;
    opticalflow(:,:,3) = u_back;
    opticalflow(:,:,4) = v_back;
end




%---------stack method mean filter(pixel by pixel)---------
function[frout] = stack1(base,top,of_u,of_v,p)
    frout = base;
%         figure;
%     subplot(211);
%     imshow(top);
%     subplot(212);
%     imshow(base);
    basegray = rgb2gray(base);
    m = mean(basegray(:))*255;
    if m < 90 % if it is night time
        for i = 1:size(base,1)
            for j = 1:size(base,2)
                if (of_u(i,j)~=0 || of_v(i,j)~=0)
                    lumda = base(i,j,1)/(base(i,j,1) + top(i,j,1));%according to red layer
                    frout(i,j,:) = base(i,j,:).*lumda + top(i,j,:).*(1-lumda);%weighted contribute (for nighttime)
                end
            end
        end
    else
        for i = 1:size(base,1)
            for j = 1:size(base,2)
                if (of_u(i,j)~=0 || of_v(i,j)~=0)
%                     frout(i,j,:) = (base(i,j,:) + top(i,j,:))/2;%quadratic contribute
                    frout(i,j,:) = (base(i,j,:)*(p-1) + top(i,j,:))/p;%equal contribute (for daytime)
                end
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
%to function properly, the frame number should be odd 
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
                num = label(1);%the first frame have median graylevel 
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
                num = label(1);%the first frame have median graylevel 
                frNum(i,j) = num;%record
                frout(i,j,:) = frlist(i,j,:,num);%assignment
            end
        end
    end
end

%---------generate method general max filter(pixel by pixel)---------
function[frout] = stack6(base,frgraylist,frlist,fgof)
    frNum = zeros(size(fgof,1),size(fgof,2));%selection record
    frout = base;
    for i = 1:size(fgof,1)
        for j = 1:size(fgof,2)
            if (fgof(i,j,1)~=0 || fgof(i,j,2)~=0)
                MAX = max(frgraylist(i,j,:));
                label = find(frgraylist(i,j,:)==MAX);
                num = label(round(size(label,1)/2));%the first frame have median graylevel 
                gfsize = min(num-1,size(frgraylist,3)-num);
                gf = fspecial('gaussian',[1 2*gfsize+1]);
                gfpad = zeros(1,size(frgraylist,3));
                gfpad(num-gfsize:num+gfsize) = gf;
                frNum(i,j) = num;%record
                R(:) = frlist(i,j,1,:);
                R = R.*gfpad;
                G(:) = frlist(i,j,2,:);
                G = G.*gfpad;
                B(:) = frlist(i,j,3,:);
                B = B.*gfpad;
                frout(i,j,1) = sum(R);%assignment R
                frout(i,j,2) = sum(G);%assignment G
                frout(i,j,3) = sum(B);%assignment B
            end
        end
    end
end