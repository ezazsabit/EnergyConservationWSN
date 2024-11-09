def getData(filename):

    with open(filename) as f:
        data = f.readlines()

    dataset = []
    label = []
    for i in range(len(data)):
        tmp = data[i].split(',')
        if tmp[-1] == 'g\n':
            label.append(1)
        else:
            label.append(0)
        tmp = [float(tmp[i]) for i in range(len(tmp) - 1)]
        dataset.append(tmp)

    return dataset, label