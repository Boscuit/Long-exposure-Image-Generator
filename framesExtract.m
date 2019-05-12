obj = VideoReader('D://EIE4512//project//realTest//test (7).mp4');
for k = 240:330
    frame = read(obj,k);
    frame = imresize(frame,0.5);
    imwrite(frame,strcat('D:\EIE4512\project\realTest\frames7\',num2str(k),'.jpg'),'jpg');
end
    