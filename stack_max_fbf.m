%---------stack method max filter(frame by frame, pixel by pixel)---------
function[frout] = stack_max_fbf(base,top,of_u,of_v,~)
    basegray = rgb2gray(base);
    topgray = rgb2gray(top);
    frout = base;
    for i = 1:size(base,1)
        for j = 1:size(base,2)
            if (of_u(i,j)~=0 || of_v(i,j)~=0)
                selection = max(basegray(i,j),topgray(i,j));
                if selection == topgray(i,j)
                    frout(i,j,:) = top(i,j,:);%assignment
                end
            end
        end
    end
end