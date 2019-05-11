clear;
excomprate = 0.5;
ofcomprate = 1;%scale rate
threshold = 1;
obj = VideoReader('D://EIE4512//project//realTest//test (8).mp4');
Num = obj.NumberOfFrame;
frselect = [120,90]; %frnumber[framestart,number]
frlist = zeros(obj.Height*excomprate,obj.Width*excomprate,3,frselect(2));
frgraylist = zeros(obj.Height*excomprate,obj.Width*excomprate,frselect(2));
frlist_b = zeros(size(frlist));
frgraylist_b = zeros(size(frgraylist));
fgof = zeros(obj.Height*excomprate,obj.Width*excomprate,2);%accumulated foreground optical flow
fgoflist = zeros(obj.Height*excomprate,obj.Width*excomprate,2,frselect(2));
%fgofNlist = zeros(obj.Height/2,obj.Width/2,frselect(2));


for k = 1:frselect(2)
    frame = read(obj,k+frselect(1)-1);
    frame = imresize(frame,excomprate);
    frlist(:,:,:,k) = im2double(frame);
    frgraylist(:,:,k) = rgb2gray(frlist(:,:,:,k));
    %imwrite(frame,strcat('D:\EIE4512\project\gtaTest1\frames\',num2str(k),'.jpg'),'jpg');
end

frout = frlist(:,:,:,1);
frlist_b(:,:,:,frselect(2)) = frlist(:,:,:,frselect(2));
frgraylist_b(:,:,frselect(2)) = frgraylist(:,:,frselect(2));
    
for p = 1:frselect(2)-1
    im1 = frgraylist(:,:,p);
    im2 = frgraylist(:,:,p+1);
    im1 = imresize(im1, ofcomprate); % rescale
    im2 = imresize(im2, ofcomprate); % rescale
    disp(['runing frame ',num2str(p+frselect(1)-1),'.'])
    [opticalflow,IDX] = getopticalflow2(im1,im2,threshold,500);
    fgoflist(:,:,:,p) = opticalflow(:,:,1:2);
    fgof = fgof + fgoflist(:,:,:,p);

    % stacking(1-2)
%     frout = stack1(frout,frlist(:,:,:,p+1),fgoflist(:,:,1,p),fgoflist(:,:,2,p),p);
%     frout = stack2(frout,frlist(:,:,:,p+1),fgoflist(:,:,1,p),fgoflist(:,:,2,p),threshold);
%     figure();
%     imshow(frout);
%     frlist_b(:,:,:,p) = getmotionblur2(frlist(:,:,:,p),IDX,opticalflow(:,:,1),opticalflow(:,:,2));
%     frgraylist_b(:,:,p) = getmotionblur2(frgraylist(:,:,p),IDX,opticalflow(:,:,1),opticalflow(:,:,2));

    %stack1
%     frout = stack1(frout,frlist(:,:,:,p),fgoflist(:,:,1,p),fgoflist(:,:,2,p),p);

    % frout = getmotionblur2(frout,IDX,fgof(:,:,1),fgof(:,:,2));

%     % downsize u and v
%     u_deci = opticalflow(1:10:end, 1:10:end, 1);
%     v_deci = opticalflow(1:10:end, 1:10:end, 2);
%     % get coordinate for u and v in the original frame
%     [m, n] = size(im1);
%     [X,Y] = meshgrid(1:n, 1:m);
%     X_deci = X(1:10/ofcomprate:end, 1:10/ofcomprate:end);
%     Y_deci = Y(1:10/ofcomprate:end, 1:10/ofcomprate:end);
%     % Plot optical flow field
%     imshow(frout);
%     hold on;
%     % draw the velocity vectors
%     quiver(X_deci, Y_deci, u_deci,v_deci, 'y')

end
    
%stack(3-5)
  frout = stack5(frout,frgraylist,frlist,fgof);


% frout = getmotionblur2(frout,IDX,fgof(:,:,1),fgof(:,:,2));
figure();
imshow(frout);

% downsize u and v
u_deci = fgof(1:10:end, 1:10:end, 1);
v_deci = fgof(1:10:end, 1:10:end, 2);
% get coordinate for u and v in the original frame
[m, n] = size(im1);
[X,Y] = meshgrid(1:n, 1:m);
X_deci = X(1:10/ofcomprate:end, 1:10/ofcomprate:end);
Y_deci = Y(1:10/ofcomprate:end, 1:10/ofcomprate:end);
% Plot optical flow field
figure();
imshow(frlist(:,:,:,1));
hold on;
% draw the velocity vectors
quiver(X_deci, Y_deci, u_deci,v_deci, 'y')



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
function [opticalflow,IDX] = getopticalflow2(im1,im2,threshold,parts)
    opticalflow = zeros([size(im1),4]);
    ww = 10;
    w = round(ww/2);
    %get superpixel and pad in boudrary with 0
    [L,N] = superpixels(im1,parts);% L superpixel-map with label 
%     figure
%     BW = boundarymask(L);
%     imshow(imoverlay(im1,BW,'cyan'),'InitialMagnification',67)

    Lpad = zeros(size(L));
    Lpad(w+1:size(im1,1)-w-1,w+1:size(im1,2)-w-1) = L(w+1:size(im1,1)-w-1,w+1:size(im1,2)-w-1);%pad in boudary with 0
    
    %get those pixels with the same label
    IDXpad = label2idx(Lpad);%for calculation
    IDX = label2idx(L);%for assignment
    IDXfinalpad = cell(1,N);% to avoid last label being padded and cell lost
    for label = 1:size(IDXpad,2)
        IDXfinalpad{label} = IDXpad{label};
    end

    % Lucas Kanade Here
    % for each point, calculate I_x, I_y, I_t
    Ix_m = conv2(im2, [-1 1; -1 1], 'valid'); % partial on x
    Iy_m = conv2(im2, [-1 -1; 1 1], 'valid'); % partial on y
    It_m = conv2(im1, ones(2), 'valid') + conv2(im2, -ones(2), 'valid'); % partial on t??
    u = zeros(size(im1));
    v = zeros(size(im1));
    u_fore = zeros(size(im1));
    v_fore = zeros(size(im1));
    u_back = zeros(size(im1));
    v_back = zeros(size(im1));
    
    % within window ww * ww
    for labelVal = 1:N
        allidxpad = IDXfinalpad{labelVal};
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
            
            u(allidx)=nu(1);
            v(allidx)=nu(2);

            %classify fore/back ground
%             if norm(nu,2)>threshold
%                 u_fore(allidx)=nu(1);
%                 v_fore(allidx)=nu(2);
%                 u_back(allidx)=0;
%                 v_back(allidx)=0;
%             else
%                 u_fore(allidx)=0;
%                 v_fore(allidx)=0;
%                 u_back(allidx)=nu(1);
%                 v_back(allidx)=nu(2);
%             end
        end
    end
    uv = (u.^2+v.^2).^0.5;
    adpth = getOtusthreshold(uv);
    disp(['adaptive threshold: ',num2str(adpth),'.'])
    for labelVal = 1:N
        allidxpad = IDXfinalpad{labelVal};
        allidx = IDX{labelVal};
        if size(allidxpad,1)~=0
            %classify fore/back ground
            if uv(allidx(1))>max(adpth,threshold)
                u_fore(allidx)=u(allidx(1));
                v_fore(allidx)=v(allidx(1));
                u_back(allidx)=0;
                v_back(allidx)=0;
            else
                u_fore(allidx)=0;
                v_fore(allidx)=0;
                u_back(allidx)=u(allidx(1));
                v_back(allidx)=v(allidx(1));
            end
        end
    end
    opticalflow(:,:,1) = u_fore;
    opticalflow(:,:,2) = v_fore;
    opticalflow(:,:,3) = u_back;
    opticalflow(:,:,4) = v_back;
    
    % downsize u and v
%     if th<0.6
%         u_deci = opticalflow(1:10:end, 1:10:end, 1);
%         v_deci = opticalflow(1:10:end, 1:10:end, 2);
%         % get coordinate for u and v in the original frame
%         [m, n] = size(im1);
%         [X,Y] = meshgrid(1:n, 1:m);
%         X_deci = X(1:10:end, 1:10:end);
%         Y_deci = Y(1:10:end, 1:10:end);
%         % Plot optical flow field
%         figure();
%         imshow(im1);
%         hold on;
%         % draw the velocity vectors
%         quiver(X_deci, Y_deci, u_deci,v_deci, 'y')
%     end
end


%--------get motion blur 1 ------------
function[fr_b] = getmotionblur1(base,IDX,of_u,of_v)
    numofpixel = size(base,1)*size(base,2);
    fr_b = base;
    N = size(IDX,2);
    for labelVal = 1:N
        allidx = IDX{labelVal};
        allidx_RGB = [allidx;allidx+numofpixel;allidx+numofpixel*2];
        px = allidx(1);
        if (of_u(px)~=0 || of_v(px)~=0)
            %get motion blur
            sita = atan(of_v(px)/of_u(px))*(180/pi);
            norm = sqrt(of_u(px)^2+of_v(px)^2);
            H = fspecial('motion',norm*10,sita);
%             Hpad = zeros(size(H));
%             start1 = round(size(H,1)*0.25);
%             end1 = round(size(H,1)*0.75);
%             start2 = round(size(H,2)*0.25);
%             end2 = round(size(H,2)*0.75);
%             Hpad(start1:end1,start2:end2) = H (start1:end1,start2:end2);%pad the corner of the filter to eliminate the boudrary effect
%             Hpad = Hpad/sum(Hpad(:));%normalize the filter
            Hpad = zeros(size(H));
            Hpad(round(size(H,1)/2):size(H,1),1:round(size(H,2)/2)) = H (round(size(H,1)/2):size(H,1),1:round(size(H,2)/2));
            Hpad = Hpad/sum(Hpad(:));%normalize the filter
            segment_b = imfilter(base,Hpad,'replicate');
            if size(base,3) == 1 %gray level image
                fr_b(allidx) = segment_b(allidx);
            else
                fr_b(allidx_RGB) = segment_b(allidx_RGB);
            end
        end
    end
%     figure;
%     imshow(fr_b);
end

%--------get motion blur 2 ------------
function[fr_b] = getmotionblur2(base,IDX,of_u,of_v)
    OS = cell(1,4); %cell for each direction after octonary segmentation
    for i = 1:size(base,1)
        for j = 1:size(base,2)
            if (of_u(i,j)~=0 || of_v(i,j)~=0)
                sita = atan(of_v(i,j)/of_u(i,j))*(180/pi);
                if (sita>=0 && sita<45)
                    OS{1} = [OS{1};i+(j-1)*size(base,1)];%add label to cell 1
                elseif (sita>=45 && sita<90)
                    OS{2} = [OS{2};i+(j-1)*size(base,1)];%add label to cell 2
                elseif (sita>=90 && sita<135)
                    OS{3} = [OS{3};i+(j-1)*size(base,1)];%add label to cell 3
                else
                    OS{4} = [OS{4};i+(j-1)*size(base,1)];%add label to cell 4
                end
            end
        end
    end
    numofpixel = size(base,1)*size(base,2);
    fr_b = base;
    for labelVal = 1:4
        if size(OS{labelVal})~=0
            allidx = OS{labelVal};
            allidx_RGB = [allidx;allidx+numofpixel;allidx+numofpixel*2];
            %get motion blur
            mean_u = mean(abs(of_u(allidx)));
            mean_v = mean(abs(of_v(allidx)));
            sita = atan(mean_v/mean_u)*(180/pi);
            norm = sqrt(mean_u^2+mean_u^2);
            H = fspecial('motion',norm*10,sita);
            Hpad = zeros(size(H));
            Hpad(round(size(H,1)/2):size(H,1),1:round(size(H,2)/2)) = H (round(size(H,1)/2):size(H,1),1:round(size(H,2)/2));
            Hpad = Hpad/sum(Hpad(:));%normalize the filter
            segment_b = imfilter(base,Hpad,'replicate');
            if size(base,3) == 1 %gray level image
                fr_b(allidx) = segment_b(allidx);
            else
                fr_b(allidx_RGB) = segment_b(allidx_RGB);
            end
        end
    end
%     figure;
%     imshow(fr_b);
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

%---------stack method max (pixel by pixel)---------
function[frout] = stack6(base,frgraylist,frlist,fgof)
    frNum = zeros(size(fgof,1),size(fgof,2));%selection record
    frout = base;
    for i = 1:size(fgof,1)
        for j = 1:size(fgof,2)
            if (fgof(i,j,1)~=0 || fgof(i,j,2)~=0)
%                 label = find(frgraylist(i,j,:)==max(frgraylist(i,j,:)));
%                 num = label(1);%the first frame have median graylevel 
                [label,num] = max(frgraylist(i,j,:));
                frNum(i,j) = num;%record
                frout(i,j,:) = frlist(i,j,:,num);%assignment
            end
        end
    end
end