function [refineim] = refineuvim(uvim)
    se = strel('disk',100);
    d = imdilate(uvim,se);
    c = imerode(d,se);
    mf = fspecial('average',[50 50]);
    cb = imfilter(c,mf,'symmetric');
    cb = cb.*2;
    for i = 1:size(uvim,1)
        for j = 1:size(uvim,2)
            if cb(i,j) > 1
                cb(i,j,1) = 1;
            end
        end
    end
    refineim = cb;
end