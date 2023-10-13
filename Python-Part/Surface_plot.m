% plot schemic diagram for optimization on simplex

% make data
x = linspace(-2,2,1000);
y = linspace(-2,2,1000);

[X,Y] = meshgrid(x,y);
Z = X.^2 + Y.^2;

for i = 1:1000
    for j = 1:1000
        if (Y(i,j) > - X(i,j)/sqrt(3) + 2/sqrt(3)) || ...
           (Y(i,j) < + X(i,j)/sqrt(3) - 2/sqrt(3)) || (X(i,j) < -1)
           X(i,j) = nan;
           Y(i,j) = nan;
           Z(i,j) = nan;
        end
    end
end
%%
s = surf(X,Y,Z);
s.EdgeColor = 'none';
view(-30,60)
grid off
axis off
set(gca, 'Color', 'None');
set(gcf, 'Color', 'None');
exportgraphics(gcf,'surface_try1.eps', ...
    'ContentType','vector', ...
    'BackgroundColor','none')