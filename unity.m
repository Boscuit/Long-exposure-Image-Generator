obj = VideoReader('D://EIE4512//project//gtaTest1//gtaTest1.mp4');
frselect = [20,5]; %frnumber[framestart,number]
frlist = zeros(obj.Height/2,obj.Width/2,3,frselect(2));
frgraylist = zeros(obj.Height/2,obj.Width/2,frselect(2));
fgof = zeros(obj.Height/2,obj.Width/2,2);%accumulated foreground optical flow
fgoflist = zeros(obj.Height/2,obj.Width/2,2,frselect(2));
%fgofNlist = zeros(obj.Height/2,obj.Width/2,frselect(2));
rate = 1;%scale rate
threshold = 0.3;

for k = 1:frselect(2)
    frame = read(obj,k+frselect(1));
    frame = imresize(frame,0.5);
    frlist(:,:,:,k) = im2double(frame);
    frgraylist(:,:,k) = rgb2gray(frlist(:,:,:,k));
    %imwrite(frame,strcat('D:\EIE4512\project\gtaTest1\frames\',num2str(k),'.jpg'),'jpg');
end

frout = frlist(:,:,:,1);
    
for p = 1:frselect(2)-1
    im1 = frgraylist(:,:,p);
    im2 = frgraylist(:,:,p+1);
    im1 = imresize(im1, rate); % rescale
    im2 = imresize(im2, rate); % rescale
    opticalflow = getopticalflow2(im1,im2,threshold);
    fgoflist(:,:,:,p) = opticalflow(:,:,1:2);
    fgof = fgof + fgoflist(:,:,:,p);

    % stacking
    %method1
%     for i = 1:size(im1,1)
%         for j = 1:size(im1,2)
%             if (fgoflist(i,j,1,p)~=0 || fgoflist(i,j,2,p)~=0)
%                 frout(i,j,:) = (frout(i,j,:) + frlist(i,j,:,p+1))/2;
%             end
%         end
%     end

%     frout = stack1(frout,frlist(:,:,:,p+1),fgoflist(:,:,1,p),fgoflist(:,:,2,p));
    

    %method2
%     fgofNlist(:,:,p) = sqrt(fgoflist(:,:,1,p).^2 + fgoflist(:,:,2,p).^2);
%     fgofNlist(:,:,p) = fgofNlist(:,:,p)/threshold;
%     fgofNlist(:,:,p) = fgofNlist(:,:,p)./(fgofNlist(:,:,p)+1);
%     frout = frout.*(1-fgofNlist(:,:,p)) +  frlist(:,:,:,p+1).*fgofNlist(:,:,p);
%     figure();
%     imshow(frout);

%    frout = stack2(frout,frlist(:,:,:,p+1),fgoflist(:,:,1,p),fgoflist(:,:,2,p),threshold);
    
%     figure();
%     imshow(frout);


end
    
%method3
frout = stack3(frout,frgraylist,frlist,fgof);
% frNum = zeros(size(fgof,1),size(fgof,2));
% for i = 1:size(fgof,1)
%     for j = 1:size(fgof,2)
%         if (fgof(i,j,1)~=0 || fgof(i,j,2)~=0)
%             label = find(frgraylist(i,j,:)==median(frgraylist(i,j,:)));
%             num = label(1);
%             frNum(i,j) = num;
%             frout(i,j,:) = frlist(i,j,:,num);
%         end
%     end
% end

% for p = 1:frselect(2)-1
%     hit = frNo==p*ones(size(fgof,1),size(fgof,1));
%     for s = 1:3
%         hit3d(:,:,s) = hit;
%     end
%     frout = frlist(:,:,:,p+1).*hit3d;
% end


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
quiver(X_deci, Y_deci, u_deci,v_deci, 'y')


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

function [opticalflow] = getopticalflow2(im1,im2,threshold)
    opticalflow = zeros([size(im1),4]);
    ww = 20;
    w = round(ww/2);
    %get superpixel and pad boudrary with 0
    [L,N] = superpixels(im1,500);
    Lpad = zeros(size(L));
    Lpad(w+1:size(im1,1)-w-1,w+1:size(im1,2)-w-1) = L(w+1:size(im1,1)-w-1,w+1:size(im1,2)-w-1);
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

    % within window ww * ww
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

function[frout] = stack1(base,top,of_u,of_v)
    frout = base;
    for i = 1:size(base,1)
        for j = 1:size(base,2)
            if (of_u(i,j)~=0 || of_v(i,j)~=0)
                frout(i,j,:) = (base(i,j,:) + top(i,j,:))/2;
            end
        end
    end
end

function[frout] = stack2(base,top,of_u,of_v,threshold)
    fgofNlist = zeros(size(base,1),size(base,2));
    fgofNlist(:,:) = sqrt(of_u.^2 + of_v.^2);
    fgofNlist(:,:) = fgofNlist(:,:)/threshold;
    fgofNlist(:,:) = fgofNlist(:,:)./(fgofNlist(:,:)+1);
    frout = base.*(1-fgofNlist(:,:)) +  top.*fgofNlist(:,:);
end

function[frout] = stack3(base,frgraylist,frlist,fgof)
    frNum = zeros(size(fgof,1),size(fgof,2));
    frout = base;
    for i = 1:size(fgof,1)
        for j = 1:size(fgof,2)
            if (fgof(i,j,1)~=0 || fgof(i,j,2)~=0)
                label = find(frgraylist(i,j,:)==median(frgraylist(i,j,:)));
                num = label(1);
                frNum(i,j) = num;
                frout(i,j,:) = frlist(i,j,:,num);
            end
        end
    end
end