%--------get motion blur 2 ------------
function[fr_b] = getmotionblur2(base,IDX,of_u,of_v)
    OS = cell(1,4); %cell for each direction after octonary segmentation
    for i = 1:size(base,1)
        for j = 1:size(base,2)
            if (of_u(i,j)~=0 || of_v(i,j)~=0)
                sita = atan(of_v(i,j)/of_u(i,j))*(180/pi);
                if (sita>=0 && sita<45)
                    OS{1} = [OS{1};i+(j-1)*size(base,1)];%add label to cell 1
                elseif (sita>=45 && sita<90)
                    OS{2} = [OS{2};i+(j-1)*size(base,1)];%add label to cell 2
                elseif (sita>=90 && sita<135)
                    OS{3} = [OS{3};i+(j-1)*size(base,1)];%add label to cell 3
                else
                    OS{4} = [OS{4};i+(j-1)*size(base,1)];%add label to cell 4
                end
            end
        end
    end
    numofpixel = size(base,1)*size(base,2);
    fr_b = base;
    for labelVal = 1:4
        if size(OS{labelVal})~=0
            allidx = OS{labelVal};
            allidx_RGB = [allidx;allidx+numofpixel;allidx+numofpixel*2];
            %get motion blur
            mean_u = mean(abs(of_u(allidx)));
            mean_v = mean(abs(of_v(allidx)));
            sita = atan(mean_v/mean_u)*(180/pi);
            norm = sqrt(mean_u^2+mean_u^2);
            H = fspecial('motion',norm*10,sita);
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