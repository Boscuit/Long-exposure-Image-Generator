obj = VideoReader('D://EIE4512//project//gtaTest1//gtaTest1-1.mp4');
for k = 10:50
    frame = read(obj,k);
    frame = imresize(frame,0.5);
    imwrite(frame,strcat('D:\EIE4512\project\gtaTest1\frames1\',num2str(k),'.jpg'),'jpg');
end
    