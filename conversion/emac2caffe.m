%% CONVERT
load('retrievalSfM30k-edgemac-vgg.mat')

%% EDGE FILTER
% Edge Detection (Dollàr and Zitnick) and Edge Filtering have to be
% implemented outside the network forward for caffe implementations.
% Edge Filtering performs:
% Y = (W .* X.^P)./(exp(-S .* (X - T)) + 1);
% with the following parameters

edge_filter_w = net.params(27).value;
edge_filter_p = net.params(28).value;
edge_filter_s = net.params(29).value;
edge_filter_t = net.params(30).value;

net = dagnn.DagNN.loadobj(net);

net.removeLayer('edgelayer')
net.removeLayer('objective')
net.removeLayer('error')
net.removeLayer('l2descriptor')

net.meta.normalization.imageSize(3) = 1;
net.meta.normalization.averageImage = 0;

net.rebuild()
net_struct = net.saveobj();
dagnn_caffe_deploy(net_struct, 'retrievalSfM30k-edgemac-vgg', 'inputBlobName', 'x0', 'outputBlobName', 'pooldescriptor', 'removeDropout', false, 'replaceSoftMaxLoss', false);

%% TEST
input = single(ones(224, 224, 1, 10));
net.eval({'x0', input});
out1 = net.getVar('pooldescriptor').value;

caffenet = caffe.Net('retrievalSfM30k-edgemac-vgg.prototxt', 'retrievalSfM30k-edgemac-vgg.caffemodel', 'test');
caffenet.blobs('x0').reshape([224,224,1,10]);
out2 = caffenet.forward({input});
out2 = out2{1};

max(max(abs(out1 - out2)))





