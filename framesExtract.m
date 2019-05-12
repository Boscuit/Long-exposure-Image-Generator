obj = VideoReader('D://EIE4512//project//realTest//test (4).mp4');
for k = 1:90
    frame = read(obj,k);
    frame = imresize(frame,0.5);
    imwrite(frame,strcat('D:\EIE4512\project\realTest\frames4\',num2str(k),'.jpg'),'jpg');
end
    