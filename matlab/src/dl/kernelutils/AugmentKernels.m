function netout = AugmentKernels( netin )
%AugmentKernels  augmentting the kernels of the input network.
%   Detailed explanation goes here

netout = netin;

nlayers = numel(netout.layers);

for i = 1:nlayers - 2
  if strcmpi(netout.layers{i}.type, 'conv')
    curkernels = netout.layers{i}.weights{1, 1};
    curweights = netout.layers{i}.weights{1, 2};
    
    augkernels = curkernels;
    augweights = curweights;
    
    netout.layers{i}.weights{1, 1} = cat(4, curkernels, augkernels);
    netout.layers{i}.weights{1, 2} = cat(2, curweights, augweights);
  end
end

end
