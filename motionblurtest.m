im = imread('D:\EIE4512\project\gtaTest1\frames2\5.jpg');
figure;
imshow(im);
frame = imresize(im,1);
figure;
imshow(frame);
imgray = rgb2gray(im);
H = fspecial('motion',50,20);
g_x=fspecial('gaussian',[3 5]);
Hpad = zeros(size(H));
start1 = round(size(H,1)*0.5);
end1 = round(size(H,1)*0.5);
start2 = round(size(H,2)*0.5);
end2 = round(size(H,2)*0.5);
Hpad(start1:size(H,1),1:end2) = H (start1:size(H,1),1:end2);
Hpad = Hpad/sum(Hpad(:));%normalize the filter
im_b = imfilter(im,g_x,'replicate');
im_b2 = imfilter(im,H,'replicate');