%---------stack method mean filter(pixel by pixel)---------
function[frout] = stack_mean_fbf(base,top,of_u,of_v,p)
    frout = base;
    basegray = rgb2gray(base);
    m = mean(basegray(:))*255;
    if m < 90 % if it is night time
        for i = 1:size(base,1)
            for j = 1:size(base,2)
                if (of_u(i,j)~=0 || of_v(i,j)~=0)
                    lumda = base(i,j,1)/(base(i,j,1) + top(i,j,1));%according to red layer
                    frout(i,j,:) = base(i,j,:).*lumda + top(i,j,:).*(1-lumda);%weighted contribute (for nighttime)
                end
            end
        end
    else
        for i = 1:size(base,1)
            for j = 1:size(base,2)
                if (of_u(i,j)~=0 || of_v(i,j)~=0)
%                     frout(i,j,:) = (base(i,j,:) + top(i,j,:))/2;%quadratic contribute
                    frout(i,j,:) = (base(i,j,:)*(p-1) + top(i,j,:))/p;%equal contribute (for daytime)
                end
            end
        end
    end
end