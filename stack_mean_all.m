%---------stack method mean filter(pixel by pixel)---------
function[frout] = stack_mean_all(frgraylist,frlist)
    frout = frlist(:,:,:,1);
    basegray = rgb2gray(frout);
    m = mean(basegray(:))*255;
    if m < 90 % if it is night time
        for i = 1:size(frout,1)
            for j = 1:size(frout,2)
                lumda(:) = frgraylist(i,j,:)./sum(frgraylist(i,j,:));%according to gray level
                %weighted contribute (for nighttime)
                frlist_r(:)=frlist(i,j,1,:);
                frlist_g(:)=frlist(i,j,2,:);
                frlist_b(:)=frlist(i,j,3,:);
                frout(i,j,1) = sum(frlist_r(:).*lumda(:));
                frout(i,j,2) = sum(frlist_g(:).*lumda(:));
                frout(i,j,3) = sum(frlist_b(:).*lumda(:));
            end
        end
    else
        for i = 1:size(frout,1)
            for j = 1:size(frout,2)
                %equal contribute (for daytime)
                frout(i,j,1) = mean(frlist(i,j,1,:));
                frout(i,j,2) = mean(frlist(i,j,2,:));
                frout(i,j,3) = mean(frlist(i,j,3,:));
            end
        end
    end
end