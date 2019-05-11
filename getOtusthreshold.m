% x=imread('rice.png');
% th = getOtusthreshold(x);
% imshow(im2bw(x,th),[]);

function [threshold] = getOtusthreshold(x)   
%    a=x;
%    subplot(211);  
%    imshow(a,[]);  
   %[count x]=imhist(a);   
   [m,n]=size(x);   
   N=m*n;   
   L = 100;
   p = double(min(x(:)));
   q = double(max(x(:)));
   seg=linspace(p,q,L);

count = zeros(1,L-1);
ua = zeros(1,L-1);
pa = zeros(1,L-1);
for i = 1:m
    for j = 1:n
        for T = 1:L-1
            if x(i,j)>seg(T) && x(i,j)<seg(T+1)
                count(T)=count(T)+1;
            end
        end
    end
end

prob = count/N;
u = 0;

for T = 1:L-1
    mid = p+(T-0.5)*(q-p)/(L-1);%middle point in the region T
    u = u+prob(T)*mid; %u finally become the overall mean
    ua(T) = u;           %ua（T）is the accumulated mean of T regions so far
end

for T = 1:L-1  
    pa(T)=sum(prob(1:T));  %pa（T）is the accumulated prob of T regions so far 
end

var = (pa./(1-pa)).*(ua./pa - u).^2;   %calculate variance for every threshold T
[y,t]=max(var);  %可以取出数组的最大值及取最大值的点
threshold = p+(t-0.5)*(q-p)/(L-1);
end