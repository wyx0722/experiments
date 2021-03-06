function [ opticalFLow ] = LoadOpticalFlow( path )


xFlowPath=[path '/x_flow/'];
yFlowPath=[path '/y_flow/'];

flowImages=dir([xFlowPath '*.jpg']);

xTemp=imread([xFlowPath flowImages(1).name]);
yTemp=imread([yFlowPath flowImages(1).name]);

xFlow=zeros(size(xTemp, 1),size(xTemp, 2), length(flowImages));
yFlow=zeros(size(yTemp, 1),size(yTemp, 2), length(flowImages));

xFlow(:, :, 1)=xTemp;
yFlow(:, :, 1)=yTemp;

for i=2:length(flowImages)
    xFlow(:, :, i) = imread([xFlowPath flowImages(i).name]);
    yFlow(:, :, i) = imread([yFlowPath flowImages(i).name]);   
end

opticalFLow=complex(xFlow, yFlow);

end

