import cv2
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def dHash(image):
    image_new=image
    h,w=image.shape
    avreage = np.mean(image_new)
    hash=[]
    h1=np.zeros([1,h*(w-1)])
    # assign 1 if preceeding pixel is greater; assign 0 otherwise.
    iter=0
    for i in range(h):
        for j in range(w-1):
            if image[i,j]>image[i,j+1]:
                h1[0][iter]=1
            iter+=1
    return h1


def read_gaze(name,shape):
    img=cv2.imread(f'TDD/Expert/gaze_map/gray/{name}',cv2.IMREAD_GRAYSCALE)
    img=cv2.resize(img,(shape[1],shape[0]))
    # img = cv2.resize(img, (81, 40))
    return img
if __name__ == "__main__":
    shape=[8,9]
    print(f"shape:{shape[0]}-{shape[1]}")
    relation = np.zeros([700, 700])
    train_list = []
    with open('train_list.txt', 'r', encoding='utf-8') as f:
        namelist = f.readlines()
        for name in namelist:
            train_list.append(name[:-1])

    threshold = 0.7
    sum = 0
    for i in tqdm(range(len(train_list))):
        for j in range(i, len(train_list)):
            img1 = read_gaze(train_list[i],shape)
            img2 = read_gaze(train_list[j],shape)
            hash1=dHash(img1)
            hash2=dHash(img2)
            similar = cosine_similarity(hash1, hash2)
            relation[i][j] = similar[0][0]
            relation[j][i] = relation[i][j]
            if (similar > threshold):
                print("{} {}: {:.2f}".format(train_list[i], train_list[j], similar[0][0]))
                sum += 1

    np.save('relation_dhash', relation)

