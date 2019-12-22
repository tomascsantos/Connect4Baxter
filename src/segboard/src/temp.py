class Island:
    
    def __init__(self, row, col, g):
        self.ROW = row
        self.COL = col
        self.G = g
        self.ISLANDS = []
        self.COUNT = 0
        self.VISITED = [[False for j in range(self.COL)]for i in range(self.ROW)] 

        
    def isSafe(self, i, j): 
        # row number is in range, column number 
        # is in range and value is 1  
        # and not yet visited 
        return (i >= 0 and i < self.ROW and 
                j >= 0 and j < self.COL and 
                not self.VISITED[i][j] and self.G[i][j]) 
        
    def DFS_clear(self, i, j): 
  
        # These arrays are used to get row and  
        # column numbers of 8 neighbours  
        # of a given cell 
        rowNbr = [-1, -1, -1,  0, 0,  1, 1, 1]; 
        colNbr = [-1,  0,  1, -1, 1, -1, 0, 1]; 
          
        # Mark this cell as visited 
        self.VISITED[i][j] = True
  
        # Recur for all connected neighbours 
        for k in range(8): 
            if self.isSafe(i + rowNbr[k], j + colNbr[k]):
                if self.G[i+rowNbr[k]][j+colNbr[k]] == 1:
                    self.G[i+rowNbr[k]][j+colNbr[k]] = 0
                    self.DFS_clear(i + rowNbr[k], j + colNbr[k]) 
                
    def clearIsland(self, x, y):
        self.VISITED = [[False for j in range(self.COL)]for i in range(self.ROW)] 
        self.G[y][x] = 0
        self.DFS_clear(y, x)
  
    def DFS(self, i, j): 
  
        # These arrays are used to get row and  
        # column numbers of 8 neighbours  
        # of a given cell 
        rowNbr = [-1, -1, -1,  0, 0,  1, 1, 1]; 
        colNbr = [-1,  0,  1, -1, 1, -1, 0, 1]; 
          
        # Mark this cell as visited 
        self.VISITED[i][j] = True
  
        # Recur for all connected neighbours 
        for k in range(8): 
            if self.isSafe(i + rowNbr[k], j + colNbr[k]): 
                self.DFS(i + rowNbr[k], j + colNbr[k]) 
  
    # The main function that returns 
    # count of islands in a given boolean 
    # 2D matrix 
    def findIslands(self): 
        # Initialize count as 0 and travese  
        # through the all cells of 
        # given matrix 
        for i in range(self.ROW): 
            for j in range(self.COL): 
                # If a cell with value 1 is not visited yet,  
                # then new island found 
                if self.VISITED[i][j] == False and self.G[i][j] != 0: 
                    # Visit all cells in this island  
                    # and increment island count
                    self.DFS(i, j)
                    x = Cell(j, i)
                    self.ISLANDS.append(x)
        self.calculateSize()
        self.VISITED = [[False for j in range(self.COL)]for i in range(self.ROW)] 
        return self.ISLANDS 
    
    def calculateSize(self):
        for cell in self.ISLANDS:
            self.VISITED = [[False for j in range(self.COL)]for i in range(self.ROW)] 
            self.COUNT = 0
            self.sizeHelp(cell.getY(), cell.getX())
            cell.setSize(self.COUNT)
        
    def sizeHelp(self, y, x):
        rowNbr = [-1, -1, -1,  0, 0,  1, 1, 1]; 
        colNbr = [-1,  0,  1, -1, 1, -1, 0, 1]; 
        self.VISITED[y][x] = True
        if self.G[y][x] == 1:
            self.COUNT += 1        
        for k in range(8): 
            if self.isSafe(y + rowNbr[k], x + colNbr[k]):
                if self.G[y+rowNbr[k]][x+colNbr[k]] == 1:
                    self.sizeHelp(y + rowNbr[k], x + colNbr[k])
                    
    def leaveLargest(self):
        maxSize = self.ISLANDS[0].getSize()
        for cell in self.ISLANDS:
            if maxSize < cell.getSize():
                maxSize = cell.getSize()
        for cell in self.ISLANDS:
            if cell.getSize() != maxSize:
                self.clearIsland(cell.getX(), cell.getY())

        #print(self.G)
        return self.G

class Cell:
    
    def __init__(self, x, y):
        self.X = x
        self.Y = y
        self.SIZE = 0
    def setSize(self, size):
        self.SIZE = size
    def getX(self):
        return self.X
    def getY(self):
        return self.Y
    def getSize(self):
        return self.SIZE

#___CONTROL____#
#test_img = read_image(IMG_DIR + '/IMG-1115.jpg')
#hsv = threshold(IMG_DIR + '/IMG-1115.jpg', 'BLUE')
#row = len(hsv)
#col = len(hsv[0])
#g = Island(row, col, hsv)
#g.findIslands()
#g.leaveLargest()