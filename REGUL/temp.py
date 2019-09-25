import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def doit(imagenb, name):
    image1 = imagenb
    image = dataset['images/test_set'][image1]
    lab = dataset['labels/test_set'][image1]
    other = np.where(dataset['labels/test_set'] == lab)[0][100]
    dest = dataset['images/test_set'][other]
    time = np.linspace(0, 1, 32*5).reshape((-1, 1, 1, 1))
    interp = np.expand_dims(image, 0) * time + np.expand_dims(dest, 0) * (1 - time)
    outs = list()
    for b in range(5):
        batch = interp[32*b:32*(b+1)]
        out = workplace.session.run(dnn[-1][:32], feed_dict={dataset.images: batch,
                                                       **dnn.deter_dict(True)})
        outs.append(out)
    outs = np.concatenate(outs, 0)
    plt.figure(figsize=(14,5))
    plt.subplot(131)
    image -= image.min()
    plt.imshow(image.transpose((1, 2, 0)) / image.max(), aspect='auto')
    plt.subplot(132)
    dest -= dest.min()
    plt.imshow(dest.transpose((1, 2, 0)) / image.max(), aspect='auto')
    plt.subplot(133)
    for c in range(10):
        plt.plot(outs[:, c], c='k')
    plt.savefig(name+'test_{}_{}.png'.format(imagenb,
                                        dataset['labels/test_set'][image1]))
    plt.close()

    image = dataset['images/train_set'][image1]
    lab = dataset['labels/train_set'][image1]
    other = np.where(dataset['labels/train_set'] == lab)[0][100]
    dest = dataset['images/train_set'][other]
    time = np.linspace(0, 1, 32*5).reshape((-1, 1, 1, 1))
    interp = np.expand_dims(image, 0) * time + np.expand_dims(dest, 0) * (1 - time)
    outs = list()
    for b in range(5):
        batch = interp[32*b:32*(b+1)]
        out = workplace.session.run(dnn[-1][:32], feed_dict={dataset.images: batch,
                                                       **dnn.deter_dict(True)})
        outs.append(out)
    outs = np.concatenate(outs, 0)
    plt.figure(figsize=(14,5))
    plt.subplot(131)
    image -= image.min()
    plt.imshow(image.transpose((1, 2, 0)) / image.max(), aspect='auto')
    plt.subplot(132)
    dest -= dest.min()
    plt.imshow(dest.transpose((1, 2, 0)) / image.max(), aspect='auto')
    plt.subplot(133)
    for c in range(10):
        plt.plot(outs[:, c], c='k')
    plt.savefig(name+'train_{}_{}.png'.format(imagenb,
                                           dataset['labels/test_set'][image1]))
    plt.close()



