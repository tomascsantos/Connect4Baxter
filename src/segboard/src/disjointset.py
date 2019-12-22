class DisjointSet:

    def __init__(self, init_arr):
        self._disjoint_set = init_arr

    def find(self, elem):
        for item in self._disjoint_set:
            if elem in item:
                return self._disjoint_set[self._disjoint_set.index(item)]
        return None
    
    def union(self,index_elem1, index_elem2):

        if index_elem1 != index_elem2 and index_elem1 is not None and index_elem2 is not None:
            self._disjoint_set[index_elem2] = self._disjoint_set[index_elem2]+self._disjoint_set[index_elem1]
            del self._disjoint_set[index_elem1]
        return self._disjoint_set
        
    def get(self):
        return self._disjoint_set

    def mine(self, row, col):
        temp = 0
        for i in range(len(self._disjoint_set)):
            if i == 0:
                continue
            if self._disjoint_set[i] != 0 and self._disjoint_set[i - 1] != 0:
                self.union(i, i - 1)
            if i + col >= len(self._disjoint_set):
                continue 
            if self._disjoint_set[i] != 0 and self._disjoint_set[i + col] != 0:
                self.union(i, i + col)
        return self.get()
##_________________________________________________________________________________________

package set


class DisjointSets:

    def __init__(self, init_arr, size):
        self.array = []
        self.SIZE = size
        for index in range(size):
            self.array[index] = -1

    def validate(vertex):
        if vertex < 0 or vertex >= self.SIZE:
            break
            #throw exception here

    def sizeOf(v1):
        return self.array[find(v1)] * -1

    def parent(v1):
        validate(v1)
        return self.array[v1]

    def connected(v1, v2):
        return find(v1) == find(v2)

    def find(vertex):
        


    def union(root1, root2):
        if (array[root2] < array[root1]):
            array[root2] += array[root1]
            array[root1] = root2
        else:
            array[root1] += array[root2]
            array[root2] = root1



if __name__ == '__main__':
    #blue = threshold(IMG_DIR + '/IMG-1115.jpg', 'BLUE')
    #print(blue)
    #print(blue.shape)


    #x = np.where(np.any(mask == 0, axis=-1))
    # non_black = np.any(blue != [0,0,0], axis=-1)

    # print_img('test',blue[non_black])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    hsv = threshold(IMG_DIR + '/IMG-1115.jpg', 'BLUE')
    print(np.any(hsv[:]))
    #Disjoint
    flattened = hsv.flatten()
    disjoint = DisjointSet(flattened)
    st = disjoint.mine(4032,3024)
    print(st)