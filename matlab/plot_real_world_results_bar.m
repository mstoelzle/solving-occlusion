%% plot real-world results from Table II in paper

% data
s = {'ETH Stairs', 'Obstacles', ...
    'Gonzen mine', 'Tenerife Planetary'};
x = categorical(s);
x = reordercats(x, s);
y = [0.05544, 0.05339, 0.018;
     0.34646, 0.34751, 0.166;
     0.14473, 0.16787, 0.026;
     0.02577, 0.02705, 0.0110];
stdev = [0., 0., 0.001;
         0., 0., 0.003;
         0., 0., 0.001;
         0., 0., 0.0002];


% set(gca,'xtick',x,'XTickLabel',s,'TickLabelInterpreter','latex');
b = bar(x, y, 'grouped');
hold on

ylabel("MSE $[m^2]$", Interpreter="latex")

% Calculate the number of groups and number of bars in each group
[ngroups,nbars] = size(y);
% Get the x coordinate of the bars
x_coords = nan(nbars, ngroups);
for i = 1:nbars
    x_coords(i,:) = b(i).XEndPoints;
end

% Plot the errorbars
% er = errorbar(x, y, stdev, stdev, Linestyle='none', Color=[0 0 0]);
er = errorbar(x_coords', y, stdev, 'k', Linestyle='none', Color=[0 0 0]);

legend("Linear Interpolation", "Navier-Stokes", "Self-supervised learning", ...
       Interpreter="latex")


hold off