fig1 = imread('D:\EIE4512\project\gtaTest1\frames\20.jpg');
fig1 = im2double(fig1);
fig2 = imread('D:\EIE4512\project\gtaTest1\frames\21.jpg');
fig2 = im2double(fig2);
frout = zeros(size(fig1));
rate = 50;

% for i = 1:size(fig1,1)
%     for j = 1:size(fig1,2)
%         frout(i,j,:) = ((fig1(i,j,:)*(100 - rate) + fig2(i,j,:)*rate))/100;
%     end
% end
frout = (fig1*(100 - rate) + fig2*rate)/100;
imshow(frout);
