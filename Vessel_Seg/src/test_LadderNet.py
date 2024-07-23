from LadderNetv65 import LadderNetv6

layers = 4
filters = 10

net = LadderNetv6(num_classes=2,layers=layers,filters=filters,inplanes=1)
print(net)
