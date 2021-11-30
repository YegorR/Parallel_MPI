import csv
import kmean
from mpi4py import MPI
import math

ATTEMPTS = 3
CLUSTERS_COUNT = 5


# Чтение данных из файла, получение максимальных и минимальных значений
def read_data(filename):
    with open(filename) as csvfile:
        result = []
        reader = csv.reader(csvfile)
        first = True
        max_vector = []
        min_vector = []
        for row in reader:
            if first:
                first = False
            else:
                result.append(
                    {'id': row[0],
                     'data': [int(row[3]), int(row[4]), int(row[5]), int(row[6])]})  # Возьмем только первые 4 параметра

                if not max_vector:
                    max_vector = result[-1]['data'].copy()
                else:
                    for i in range(len(result[-1]['data'])):
                        if result[-1]['data'][i] > max_vector[i]:
                            max_vector[i] = result[-1]['data'][i]

                if not min_vector:
                    min_vector = result[-1]['data'].copy()
                else:
                    for i in range(len(result[-1]['data'])):
                        if result[-1]['data'][i] < min_vector[i]:
                            min_vector[i] = result[-1]['data'][i]
        return {'result': result, 'min_vector': min_vector, 'max_vector': max_vector}


# Нормализация данных
def normalize(data, min_vector, max_vector):
    def norm(x):
        result = []
        for i in range(len(x['data'])):
            result.append((x['data'][i] - min_vector[i]) / (max_vector[i] - min_vector[i]))
        return {'id': x['id'], 'data': result}

    return list(map(norm, data))


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Считываем данные и нормализуем
    norm_data = []
    if rank == 0:
        data = read_data('data\\data.csv')
        norm_data = normalize(data['result'], data['min_vector'], data['max_vector'])

    # Отправка нормализованных данных всем процессам
    norm_data = comm.bcast(norm_data, root=0)

    # Кластеризация и оценка путем рассчета индекса
    total_result = []
    min_index = math.inf
    for attempt in range(ATTEMPTS):
        result = kmean.kmean(norm_data, CLUSTERS_COUNT)
        index = kmean.vnnd(result)
        if index < min_index:
            total_result = result
            min_index = index

    # Собираем лучшие индексы у корневого процесса
    best_indexes = comm.gather(min_index, root=0)
    best_index_idx = 0
    if rank == 0:
        print(best_indexes)
        best_index_idx = min(range(len(best_indexes)), key=best_indexes.__getitem__)

    # Сообщаем всем процессам, какой процесс посчитал лучше всех
    best_index_idx = comm.bcast(best_index_idx, root=0)
    if rank == best_index_idx:
        print("best clusterization - ", best_index_idx)
        print("best index - ", min_index)

        with open('out.txt', 'w') as f:
            print(total_result, file=f)
