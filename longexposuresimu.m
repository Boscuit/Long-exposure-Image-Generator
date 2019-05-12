clear;
%---------parameter setting---------
excomprate = 1;
threshold = 10*excomprate;
testNum = 12;
superpixel = 500;
frselect = [224,10];%frnumber[framestart,number]test4[60,60]test7[240,90]test8[120,90]
path = strcat('D://EIE4512//project//realTest//test (',num2str(testNum),').mp4');
obj = VideoReader(path);
Num = obj.NumberOfFrame;
frselect = [frselect(1), min(frselect(2),Num-frselect(1)) ]; %bound

%---------initialization---------
frlist = zeros(obj.Height*excomprate,obj.Width*excomprate,3,frselect(2));
frgraylist = zeros(obj.Height*excomprate,obj.Width*excomprate,frselect(2));
fgof = zeros(obj.Height*excomprate,obj.Width*excomprate,2);%accumulated foreground optical flow
fgoflist = zeros(obj.Height*excomprate,obj.Width*excomprate,2,frselect(2));


for k = 1:frselect(2)
    frame = read(obj,k+frselect(1)-1);
    frame = imresize(frame,excomprate);
    frlist(:,:,:,k) = im2double(frame);
    frgraylist(:,:,k) = rgb2gray(frlist(:,:,:,k));
end

frout = frlist(:,:,:,1);
    
for p = 1:frselect(2)-1
    im1 = frgraylist(:,:,p);
    im2 = frgraylist(:,:,p+1);
    disp(['runing frame ',num2str(p+frselect(1)-1),'.'])
    [opticalflow,adpth] = getopticalflow_sp(im1,im2,threshold,superpixel);
    fgoflist(:,:,:,p) = opticalflow(:,:,1:2);
    fgof = fgof + fgoflist(:,:,:,p);

    % ------------stacking(stack_mean_fbf,stack_max_fbf)----------------
%     frout = stack_mean_fbf(frout,frlist(:,:,:,p+1),fgoflist(:,:,1,p),fgoflist(:,:,2,p),p);

    % downsize u and v
%     u_deci = opticalflow(1:10:end, 1:10:end, 1);
%     v_deci = opticalflow(1:10:end, 1:10:end, 2);
%     % get coordinate for u and v in the original frame
%     [m, n] = size(im1);
%     [X,Y] = meshgrid(1:n, 1:m);
%     X_deci = X(1:10:end, 1:10:end);
%     Y_deci = Y(1:10:end, 1:10:end);
%     % Plot optical flow field
%     figure;
%     imshow(frlist(:,:,:,p+1));
%     title(strcat('frame: ',num2str(p+frselect(1)-1),', adpth: ',num2str(adpth)));
%     hold on;
%     % draw the velocity vectors
%     quiver(X_deci, Y_deci, u_deci,v_deci, 'y')
end
    
% -----------stacking(stack_median,stack_min,stack_max)-----------
[frout,frout_only] = stack_max(frout,frgraylist,frlist,fgof);

% frout = edgefeather(frout,frout_only,5*excomprate*ofcomprate);%feather
imwrite(frout,strcat('D:\EIE4512\project\realTest\test',num2str(testNum),'result\',num2str(excomprate),'x',num2str(frselect(1)),'_',num2str(frselect(1)+frselect(2)-1),'_',num2str(superpixel),'sp',num2str(threshold),'thmax','.jpg'),'jpg');

figure();
imshow(frout);

% downsize u and v
u_deci = fgof(1:10:end, 1:10:end, 1);
v_deci = fgof(1:10:end, 1:10:end, 2);
% get coordinate for u and v in the original frame
[m, n] = size(im1);
[X,Y] = meshgrid(1:n, 1:m);
X_deci = X(1:10:end, 1:10:end);
Y_deci = Y(1:10:end, 1:10:end);
% Plot optical flow field
figure();
imshow(frlist(:,:,:,1));
hold on;
% draw the velocity vectors
quiver(X_deci, Y_deci, u_deci,v_deci, 'y')



%---------get optical flow pixel by pixel---------
function [opticalflow] = getopticalflow_pbp(im1,im2,threshold)
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
function [opticalflow,adpth] = getopticalflow_sp(im1,im2,threshold,parts)
    opticalflow = zeros([size(im1),4]);
    ww = 10;
    w = round(ww/2);
    %get superpixel and pad in boudrary with 0
    [L,N] = superpixels(im1,parts);% L superpixel-map with label 
    Lpad = zeros(size(L));
    Lpad(w+1:size(im1,1)-w-1,w+1:size(im1,2)-w-1) = L(w+1:size(im1,1)-w-1,w+1:size(im1,2)-w-1);%pad in boudary with 0
    
    %get those pixels which have the same label
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
    
    % within window ww * ww
    for labelVal = 1:N
        allidxpad = IDXfinalpad{labelVal};
        allidx = IDX{labelVal};
        if size(allidxpad,1)~=0
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
        end
    end
    uv = (u.^2+v.^2).^0.5;
    adpth = getOtusthreshold(uv,IDX);
    disp(['adaptive threshold: ',num2str(adpth),'.'])
    for labelVal = 1:N
        allidxpad = IDXfinalpad{labelVal};
        allidx = IDX{labelVal};
        if size(allidxpad,1)~=0
            %classify fore/back ground
            if uv(allidx(1))>max(adpth,threshold)
                u_fore(allidx)=u(allidx(1));
                v_fore(allidx)=v(allidx(1));
            else
                u_fore(allidx)=0;
                v_fore(allidx)=0;
            end
        end
    end
    opticalflow(:,:,1) = u_fore;
    opticalflow(:,:,2) = v_fore;
end

%---------frame by frame stack-----------
%---------stack method mean filter(pixel by pixel)---------
function[frout] = stack_mean_fbf(base,top,fgof_u,fgof_v,p)
    frout = base;
    basegray = rgb2gray(base);
    m = mean(basegray(:))*255;
    if m < 90 % if it is night time
        for i = 1:size(base,1)
            for j = 1:size(base,2)
                if (fgof_u(i,j)~=0 || fgof_v(i,j)~=0)
                    lumda = base(i,j,1)/(base(i,j,1) + top(i,j,1));%according to red layer
                    frout(i,j,:) = base(i,j,:).*lumda + top(i,j,:).*(1-lumda);%weighted contribute (for nighttime)
                end
            end
        end
    else
        for i = 1:size(base,1)
            for j = 1:size(base,2)
                if (fgof_u(i,j)~=0 || fgof_v(i,j)~=0)
%                     frout(i,j,:) = (base(i,j,:) + top(i,j,:))/2;%quadratic contribute
                    frout(i,j,:) = (base(i,j,:)*(p-1) + top(i,j,:))/p;%equal contribute (for daytime)
                end
            end
        end
    end
end

%---------stack method max filter(frame by frame, pixel by pixel)---------
function[frout] = stack_max_fbf(base,top,fgof_u,fgof_v,~)
    basegray = rgb2gray(base);
    topgray = rgb2gray(top);
    frout = base;
    for i = 1:size(base,1)
        for j = 1:size(base,2)
            if (fgof_u(i,j)~=0 || fgof_v(i,j)~=0)
                selection = max(basegray(i,j),topgray(i,j));
                if selection == topgray(i,j)
                    frout(i,j,:) = top(i,j,:);%assignment
                end
            end
        end
    end
end



%---------overall stack-------------
%---------stack method median filter(pixel by pixel)---------
%to function properly, the frame number should be odd 
function[frout,frout_only] = stack_median(base,frgraylist,frlist,fgof)
    frNum = zeros(size(fgof,1),size(fgof,2));%selection record
    frout = base;
    frout_only = zeros(size(base));
    for i = 1:size(fgof,1)
        for j = 1:size(fgof,2)
            if (fgof(i,j,1)~=0 || fgof(i,j,2)~=0)
                label = find(frgraylist(i,j,:)==median(frgraylist(i,j,:)));
                num = label(1);%the first frame have median graylevel 
                frNum(i,j) = num;%record
                frout(i,j,:) = frlist(i,j,:,num);%assignment
                frout_only(i,j,:) = frlist(i,j,:,num);%assignment
            end
        end
    end
end

%---------stack method min filter(pixel by pixel)---------
function[frout,frout_only] = stack_min(base,frgraylist,frlist,fgof)
    frNum = zeros(size(fgof,1),size(fgof,2));%selection record
    frout = base;
    frout_only = zeros(size(base));
    for i = 1:size(fgof,1)
        for j = 1:size(fgof,2)
            if (fgof(i,j,1)~=0 || fgof(i,j,2)~=0)
                label = find(frgraylist(i,j,:)==min(frgraylist(i,j,:)));
                num = label(1);%the first frame have median graylevel 
                frNum(i,j) = num;%record
                frout(i,j,:) = frlist(i,j,:,num);%assignment
                frout_only(i,j,:) = frlist(i,j,:,num);%assignment
            end
        end
    end
end

%---------stack method max filter(pixel by pixel)---------
function[frout,frout_only] = stack_max(base,frgraylist,frlist,fgof)
    frNum = zeros(size(fgof,1),size(fgof,2));%selection record
    frout = base;
    frout_only = zeros(size(base));
    for i = 1:size(fgof,1)
        for j = 1:size(fgof,2)
            if (fgof(i,j,1)~=0 || fgof(i,j,2)~=0)
                label = find(frgraylist(i,j,:)==max(frgraylist(i,j,:)));
                num = label(1);%the first frame have median graylevel 
                frNum(i,j) = num;%record
                frout(i,j,:) = frlist(i,j,:,num);%assignment
                frout_only(i,j,:) = frlist(i,j,:,num);%assignment
            end
        end
    end
end

