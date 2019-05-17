obj = VideoReader('D://EIE4512//project//realTest//test (11).mp4');
for k = 200:299
    frame = read(obj,k);
    frame = imresize(frame,0.5);
    imwrite(frame,strcat('D:\EIE4512\project\realTest\frames11\',num2str(k),'.jpg'),'jpg');
end
    