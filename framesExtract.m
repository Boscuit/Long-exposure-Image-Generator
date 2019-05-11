obj = VideoReader('D://EIE4512//project//realTest//test (8).mp4');
for k = 120:200
    frame = read(obj,k);
    frame = imresize(frame,0.5);
    imwrite(frame,strcat('D:\EIE4512\project\realTest\frames8\',num2str(k),'.jpg'),'jpg');
end
    