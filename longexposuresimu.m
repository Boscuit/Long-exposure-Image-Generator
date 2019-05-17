clear;
%---------parameter setting---------
excomprate = 1;
fuse = 25*excomprate;%bound the extreme optical flow
testNum = 11;
superpixel = 500;
frselect = [2,100];%frnumber[framestart,number]test1[300,50]test3[2,100]test4[60,60]test7[280,50]test8[120,90]test11[2,100]test12[200,100]
path = strcat('D://EIE4512//project//realTest//test (',num2str(testNum),').mp4');
obj = VideoReader(path);
Num = obj.NumberOfFrame;
frselect = [frselect(1), min(frselect(2),Num-frselect(1)) ]; %bound

%---------initialization---------
frlist = zeros(obj.Height*excomprate,obj.Width*excomprate,3,frselect(2));
frgraylist = zeros(obj.Height*excomprate,obj.Width*excomprate,frselect(2));
of = zeros(obj.Height*excomprate,obj.Width*excomprate,2);%accumulated optical flow


for k = 1:frselect(2)
    frame = read(obj,k+frselect(1)-1);
    frame = imresize(frame,excomprate);
    frlist(:,:,:,k) = im2double(frame);
    frgraylist(:,:,k) = rgb2gray(frlist(:,:,:,k));
end

frout = frlist(:,:,:,frselect(2));
    
for p = 1:frselect(2)-1
    im1 = frgraylist(:,:,p);
    im2 = frgraylist(:,:,p+1);
    disp(['runing frame ',num2str(p+frselect(1)-1),'.'])
    [opticalflow] = getopticalflow_sp(im1,im2,fuse,superpixel); % superpixel
%     [opticalflow] = getopticalflow_pbp(im1,im2); % pixel to pixel
    of = of + opticalflow;
end
    
uv2 = of(:,:,1).^2+of(:,:,2).^2;
max = max(uv2(:));
median = median(uv2(:));
alpha = max*0.1+median*0.9;%!!!!!!key point of effect
uvim = uv2./alpha;
uvimr = refineuvim(uvim);%refine uvim

th = graythresh(uvim);%see value over 1 as 1
uvimbw=imbinarize(uvim,th);

midfr = frlist(:,:,:,round(frselect(2)/2));
%Other function to used for different effect:[stack_min_all,stack_median_all,stack_mean_all,]
froutmax= stack_max_all(frgraylist,frlist);
frout = midfr.*(1-uvimr)+froutmax.*uvimr;



% imwrite(uvim,strcat('D:\EIE4512\project\realTest\test',num2str(testNum),'result\N',num2str(excomprate),'x',num2str(frselect(1)),'_',num2str(frselect(1)+frselect(2)-1),'_',num2str(superpixel),'sp',num2str(fuse),'fuse_uvim','.jpg'),'jpg');
% imwrite(uvimr,strcat('D:\EIE4512\project\realTest\test',num2str(testNum),'result\N',num2str(excomprate),'x',num2str(frselect(1)),'_',num2str(frselect(1)+frselect(2)-1),'_',num2str(superpixel),'sp',num2str(fuse),'fuse_uvimr','.jpg'),'jpg');
% imwrite(uvimbw,strcat('D:\EIE4512\project\realTest\test',num2str(testNum),'result\N',num2str(excomprate),'x',num2str(frselect(1)),'_',num2str(frselect(1)+frselect(2)-1),'_',num2str(superpixel),'sp',num2str(fuse),'fuse_uvimbw','.jpg'),'jpg');
% imwrite(froutmax,strcat('D:\EIE4512\project\realTest\test',num2str(testNum),'result\N',num2str(excomprate),'x',num2str(frselect(1)),'_',num2str(frselect(1)+frselect(2)-1),'_',num2str(superpixel),'sp',num2str(fuse),'fuse_max_all','.jpg'),'jpg');
% imwrite(frout,strcat('D:\EIE4512\project\realTest\test',num2str(testNum),'result\N',num2str(excomprate),'x',num2str(frselect(1)),'_',num2str(frselect(1)+frselect(2)-1),'_',num2str(superpixel),'sp',num2str(fuse),'fuse_max_refined','.jpg'),'jpg');

% figure();
% imshow(frout);title('frout');


% fgof=of.*uvimbw;
% % downsize u and v
% u_deci = fgof(1:10:end, 1:10:end, 1);
% v_deci = fgof(1:10:end, 1:10:end, 2);
% % get coordinate for u and v in the original frame
% [m, n] = size(im1);
% [X,Y] = meshgrid(1:n, 1:m);
% X_deci = X(1:10:end, 1:10:end);
% Y_deci = Y(1:10:end, 1:10:end);
% % Plot optical flow field
% figure();
% imshow(frlist(:,:,:,1));
% hold on;
% % draw the velocity vectors
% quiver(X_deci, Y_deci, u_deci,v_deci, 'y')














