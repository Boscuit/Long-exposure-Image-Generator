%--------get motion blur 1 ------------
function[fr_b] = getmotionblur1(base,IDX,of_u,of_v)
    numofpixel = size(base,1)*size(base,2);
    fr_b = base;
    N = size(IDX,2);
    for labelVal = 1:N
        allidx = IDX{labelVal};
        allidx_RGB = [allidx;allidx+numofpixel;allidx+numofpixel*2];
        px = allidx(1);
        if (of_u(px)~=0 || of_v(px)~=0)
            %get motion blur
            sita = atan(of_v(px)/of_u(px))*(180/pi);
            norm = sqrt(of_u(px)^2+of_v(px)^2);
            H = fspecial('motion',norm*10,sita);
%             Hpad = zeros(size(H));
%             start1 = round(size(H,1)*0.25);
%             end1 = round(size(H,1)*0.75);
%             start2 = round(size(H,2)*0.25);
%             end2 = round(size(H,2)*0.75);
%             Hpad(start1:end1,start2:end2) = H (start1:end1,start2:end2);%pad the corner of the filter to eliminate the boudrary effect
%             Hpad = Hpad/sum(Hpad(:));%normalize the filter
            Hpad = zeros(size(H));
            Hpad(round(size(H,1)/2):size(H,1),1:round(size(H,2)/2)) = H (round(size(H,1)/2):size(H,1),1:round(size(H,2)/2));
            Hpad = Hpad/sum(Hpad(:));%normalize the filter
            segment_b = imfilter(base,Hpad,'replicate');
            if size(base,3) == 1 %gray level image
                fr_b(allidx) = segment_b(allidx);
            else
                fr_b(allidx_RGB) = segment_b(allidx_RGB);
            end
        end
    end
%     figure;
%     imshow(fr_b);
end