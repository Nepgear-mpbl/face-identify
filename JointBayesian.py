import numpy as np
import pickle


def load_jbdata():
    with open('data/JBdataF.pkl', 'rb') as f:
        data = pickle.load(f)
        label = pickle.load(f)
        return data, label


def data_to_pkl(data, file_path):
    print("Saving data to file(%s). " % (file_path))
    with open(file_path, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        return True
    print("Occur Error while saving...")
    return False


def JointBayesian_Train(trainingset, label, fold="./"):
    if fold[-1] != '/':
        fold += '/'
    print(trainingset.shape)
    # the total num of image n
    n_image = len(label)
    # the dim of features  120
    n_dim = trainingset.shape[1]
    # filter the complicate label,for count the total people num
    classes, labels = np.unique(label, return_inverse=True)
    # the total people num
    n_class = len(classes)
    # save each people items
    cur = {}
    withinCount = 0
    # record the count of each people
    numberBuff = np.zeros(n_image)
    maxNumberInOneClass = 0
    for i in range(n_class):
        # get the item of i
        cur[i] = trainingset[labels == i]
        # get the number of the same label persons
        n_same_label = cur[i].shape[0]
        if n_same_label > 1:
            withinCount += n_same_label
        if numberBuff[n_same_label] == 0:
            numberBuff[n_same_label] = 1
            maxNumberInOneClass = max(maxNumberInOneClass, n_same_label)
    print("prepare done, maxNumberInOneClass=", maxNumberInOneClass)

    u = np.zeros([n_dim, n_class])
    ep = np.zeros([n_dim, withinCount])
    nowp = 0
    for i in range(n_class):
        # the mean of cur[i]
        u[:, i] = np.mean(cur[i], 0)
        b = u[:, i].reshape(n_dim, 1)
        n_same_label = cur[i].shape[0]
        if n_same_label > 1:
            ep[:, nowp:nowp + n_same_label] = cur[i].T - b
            nowp += n_same_label

    Su = np.cov(u.T, rowvar=0)
    print(Su)
    Sw = np.cov(ep.T, rowvar=0)
    print(Sw)
    oldSw = Sw
    SuFG = {}
    SwG = {}
    convergence = 1
    min_convergence = 1
    for l in range(500):
        F = np.linalg.pinv(Sw)
        u = np.zeros([n_dim, n_class])
        ep = np.zeros([n_dim, n_image])
        nowp = 0
        for mi in range(maxNumberInOneClass + 1):
            if numberBuff[mi] == 1:
                # G = −(mS μ + S ε )−1*Su*Sw−1
                temp1=mi * Su + Sw
                print(temp1.shape)
                temp2=np.linalg.pinv(temp1)
                G = -np.dot(np.dot(temp2, Su), F)
                # Su*(F+mi*G) for u
                SuFG[mi] = np.dot(Su, (F + mi * G))
                # Sw*G for e
                SwG[mi] = np.dot(Sw, G)
        for i in range(n_class):
            ##print l, i
            nn_class = cur[i].shape[0]
            # formula 7 in suppl_760
            u[:, i] = np.sum(np.dot(SuFG[nn_class], cur[i].T), 1)
            # formula 8 in suppl_760
            ep[:, nowp:nowp + nn_class] = cur[i].T + np.sum(np.dot(SwG[nn_class], cur[i].T), 1).reshape(n_dim, 1)
            nowp = nowp + nn_class

        Su = np.cov(u.T, rowvar=0)
        Sw = np.cov(ep.T, rowvar=0)
        convergence = np.linalg.norm(Sw - oldSw) / np.linalg.norm(Sw)
        print("Iterations-" + str(l) + ": " + str(convergence))
        if convergence < 1e-6:
            print("Convergence: ", l, convergence)
            break
        oldSw = Sw

        if convergence < min_convergence:
            min_convergence = convergence
            F = np.linalg.pinv(Sw)
            G = -np.dot(np.dot(np.linalg.pinv(2 * Su + Sw), Su), F)
            A = np.linalg.pinv(Su + Sw) - (F + G)
            print("min convergence:" + str(min_convergence))
            data_to_pkl(G, fold + "G.pkl")
            data_to_pkl(A, fold + "A.pkl")

    F = np.linalg.pinv(Sw)
    G = -np.dot(np.dot(np.linalg.pinv(2 * Su + Sw), Su), F)
    A = np.linalg.pinv(Su + Sw) - (F + G)
    print('no min convergence')
    data_to_pkl(G, fold + "G_con.pkl")
    data_to_pkl(A, fold + "A_con.pkl")

    return A, G


def data_pre(data):
    data = np.sqrt(data)
    data = np.divide(data, np.repeat(np.sum(data, 1), data.shape[1]).reshape(data.shape[0], data.shape[1]))
    return data


if __name__ == '__main__':
    data, label = load_jbdata()
    # print(data[0].shape)
    # dataF = data_pre(data)
    # print(dataF[0])
    # print(dataF.shape)
    # print(label.shape)
    JointBayesian_Train(data, label, 'JointBayesian_Model/')
