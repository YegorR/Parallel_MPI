import random
import math

MAX_ITERATE = 100000

# Квадрат расстояния
def distance_2(x, y):
    result = 0
    for i in range(len(x)):
        result += (x[i] - y[i]) ** 2
    return result

# Кластеризация методов k-means
def kmean(data, c):
    random.seed()
    dim = len(data[0]['data'])
    centers = []
    for i in range(c):
        center = []
        flag = True
        while flag:
            random_point = data[random.randint(0, len(data))]['data']
            if random_point not in centers:
                center = random_point
                flag = False
        centers.append(center)

    finish = False
    clusters = []
    iteration = 0
    while not finish and iteration < MAX_ITERATE:
        iteration += 1
        new_clusters = [[] for _ in range(c)]

        for item in data:
            best_c = -1
            min_distance = math.inf
            for i in range(c):
                dist = distance_2(item['data'], centers[i])
                if dist < min_distance:
                    min_distance = dist
                    best_c = i
            new_clusters[best_c].append(item)

        new_centers = []
        for i in range(c):
            new_center = [0 for _ in range(dim)]
            for item in new_clusters[i]:
                for j in range(dim):
                    new_center[j] += item['data'][j]
            if len(new_clusters[i]) == 0:
                new_center = [random.random() for _ in range(dim)]
            else:
                for j in range(dim):
                    new_center[j] /= len(new_clusters[i])
            new_centers.append(new_center)

        finish = centers == new_centers
        centers = new_centers
        clusters = new_clusters

    return {'centers': centers, 'clusters': clusters}

# Расчет индекса VNND
def vnnd(result):
    c = len(result['clusters'])
    dmin = []
    for i in range(c):
        dmin_c = [math.inf for _ in range(len(result['clusters'][i]))]
        for j in range(len(result['clusters'][i])):
            for k in range(len(result['clusters'][i])):
                if j == k:
                    continue
                distance = math.sqrt(distance_2(result['clusters'][i][j]['data'], result['clusters'][i][k]['data']))
                dmin_c[j] = min(dmin_c[j], distance)
        dmin.append(dmin_c)

    dmin_clust = [sum(dmin[i]) / len(dmin[i]) for i in range(c)]

    v = []
    for i in range(c):
        s = sum([(dmin[i][j] - dmin_clust[i])**2 for j in range(len(dmin[i]))])
        v.append(s / (len(dmin[i]) - 1))

    return sum(v)


