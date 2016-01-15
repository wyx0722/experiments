function normTraj= norm_disp_traj(traj)

normTraj=zeros(size(traj));

for i=1:size(traj, 1)

normTraj(i, 1:2:end)=traj(i, 1:2:end)./(sum(traj(i, 1:2:end)));
normTraj(i, 2:2:end)=traj(i, 2:2:end)./(sum(traj(i, 2:2:end)));

    
end
