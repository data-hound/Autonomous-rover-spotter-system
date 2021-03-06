import numpy as np
import cv2
import math
import time
cellsPerLevel=[1,6,12,24,24,24,48]
## Reads image in HSV format. Accepts filepath as input argument and returns the HSV
## equivalent of the image.
def readImageHSV(filePath):
    mazeImg = cv2.imread(filePath)
    hsvImg = cv2.cvtColor(mazeImg, cv2.COLOR_BGR2HSV)
    return hsvImg

## Reads image in binary format. Accepts filepath as input argument and returns the binary
## equivalent of the image.
def readImageBinary(filePath):
    mazeImg = cv2.imread(filePath)
    grayImg = cv2.cvtColor(mazeImg, cv2.COLOR_BGR2GRAY)
    ret,binaryImage = cv2.threshold(grayImg,200,255,cv2.THRESH_BINARY)
    return binaryImage

##  Returns sine of an angle.
def sine(angle):
    return math.sin(math.radians(angle))

##  Returns cosine of an angle
def cosine(angle):
    return math.cos(math.radians(angle))

#This function takes rectangular values and image, and returns polar coordinates with centre as origin
def rectToPol(img,X,Y):
    centre=(img[0].size/2,img[1].size/2)
    sqradius=((X-centre[0])**2+(Y-centre[1])**2)
    radius=math.sqrt(sqradius)
    

    theta_rad=math.atan2(X-centre[0],Y-centre[1])
   
    theta=math.degrees(theta_rad)
    
    return radius,theta

#This function takes polar coordinates and image and converts them to rectangular coordinates
def polToRect(img,R,Theta):
    centre=(img[0].size/2,img[1].size/2)
    X=centre[0]+R*sine(Theta)
    Y=centre[1]+R*cosine(Theta)

    return math.trunc(X),math.trunc(Y)


#This function returns the pixel coordinates of the centre of the given cell
def getCent(img,level,cellnum):
    height=img[0].size
    breadth=img[1].size

    centre=(breadth/2,height/2)
    R=(level+0.5)*40
    Theta=(cellnum-0.5)*(360.0/cellsPerLevel[level])

    sinTheta=sine(Theta)
    cosTheta=cosine(Theta)

    cent_X=centre[0]+R*sinTheta
    cent_Y=centre[1]+R*cosTheta
    
    return cent_X,cent_Y,R,Theta


#This fumction receives the current position and the theta in degrees and shifts the pointer a given angle           is the fraction of cells to travel 
def angularTravel(radius, angle, theta,img):
    height=img[0].size
    breadth=img[1].size

    centre=(breadth/2,height/2)
    angle=angle+theta
    

    sinAngle=sine(angle)
    cosAngle=cosine(angle)

    X_pos=centre[0]+radius*sinAngle
    Y_pos=centre[1]+radius*cosAngle
    

    return (math.trunc(X_pos),math.trunc(Y_pos))

#This function receives the radius and theta and a distance x, and shifts the pointer x distance in the given angle
def radialTravel(radius, angle,x,img):
    height=img[0].size
    breadth=img[1].size

    centre=(height/2,breadth/2)

    radius=radius+x

    cosAngle=cosine(angle)
    sinAngle=sine(angle)

    X_pos=centre[0]+radius*sinAngle
    Y_pos=centre[1]+radius*cosAngle

    return (math.trunc(X_pos),math.trunc(Y_pos))

#this function receives a possibly invalid level and cell combination and return a valid combinaion
def getCell(level,cellnum,size):

    if level<0:
        level=0
    if size==1 and level>4:
        level=4
    elif size==2 and level >6:
        level=6
        
    if level==0 and cellnum!=0:
        cellnum=0
    if level==1 and (cellnum>6 or cellnum<1):
        if cellnum>6:
            cellnum=cellnum%6
        if cellnum<1:
            cellnum = 6-cellnum
    if level==2 and (cellnum>12 or cellnum<1):
        if cellnum>12:
            cellnum=cellnum%12
        if cellnum<1:
            cellnum = 12-cellnum
    
    if (level==5 or level==3 or level==4) and (cellnum>24 or cellnum<1):
        if cellnum>24:
            cellnum=cellnum%24
        if cellnum<1:
            cellnum=24-cellnum
    if level==6 and (cellnum>48 or cellnum<1):
        if cellnum>48:
            cellnum=cellnum%48
        if cellnum<1:
            cellnum=48-cellnum

    return (level,cellnum)

#This function takes an image and the pixel values and determines the cell in which the pixel lies

def getCell2(img,row,col):
    breadth=img[0].size
    height=img[1].size

    centre=(breadth/2,height/2)

    

    x=row-centre[0]
    y=col-centre[1]

    r=math.sqrt(x**2+y**2)
    theta=math.atan(float(x/y))

    level=r/40
    cellnum=theta/(cellsPerLevel[int(round(level))])

    
    return (int(round(level)),int(round(cellnum)))



##  This function accepts the img, level and cell number of a particular cell and the size of the maze as input
##  arguments and returns the list of cells which are traversable from the specified cell.
def findNeighbours(img, level, cellnum, size):
    neighbours = []
    ############################# Add your Code Here ################################
    if level==0 and cellnum==0:
        R=40
        Theta=30
        for i in range(0,6):
            if img[angularTravel(R,Theta,60*i,img)]!=0:
                neighbours.append((level+1,cellnum+i+1))
                

        return neighbours
    cent_x,cent_y,R,Theta=getCent(img,level,cellnum)
    angular_cell=360.0/cellsPerLevel[level]
    radial_cell=40
    height=img[0].size
    breadth=img[1].size

    centre=(breadth/2,height/2)
    

    if img[angularTravel(R,Theta,angular_cell/2,img)]!=0:
        neighbours.append(getCell(level,cellnum+1,size))
        

    if img[angularTravel(R,Theta,-angular_cell/2,img)]!=0:
        neighbours.append(getCell(level,cellnum-1,size))


    if img[radialTravel(R,Theta,radial_cell/2,img)]==0:
        if level==1 or level==2 or level==5:
            temp=radialTravel(R,Theta,radial_cell/2,img)
            temp=rectToPol(img,temp[0],temp[1])
            
            if img[angularTravel(temp[0],temp[1],-angular_cell/4,img)]!=0:
                neighbours.append(getCell(level+1,2*cellnum-1,size))
         
            if img[angularTravel(temp[0],temp[1],angular_cell/4,img)]!=0:
                neighbours.append(getCell(level+1,2*cellnum,size))
          
        else:
            temp=radialTravel(R,Theta,radial_cell/2,img)
            temp=rectToPol(img,temp[0],temp[1])
            if img[angularTravel(temp[0],temp[1],-angular_cell/4,img)]!=0:
                neighbours.append(getCell(level+1,cellnum,size))
           
            if img[angularTravel(temp[0],temp[1],angular_cell/4,img)]!=0:
                neighbours.append(getCell(level+1,cellnum+1,size))
            

        
       
    if img[radialTravel(R,Theta,-radial_cell/2,img)]!=0:
         if level==5 or level==4:
             neighbours.append(getCell(level-1,cellnum,size))
             
         else:
             cellnum_=cellnum/2 if cellnum%2==0 else cellnum/2+1
             neighbours.append(getCell(level-1,cellnum_,size))
             
        
        
    if img[radialTravel(R,Theta,radial_cell/2,img)]!=0:
        if level==3 or level==4 :
            neighbours.append(getCell(level+1,cellnum,size))



    #################################################################################
    return neighbours

##  colourCell function takes 5 arguments:-
##            img - input image
##            level - level of cell to be coloured
##            cellnum - cell number of cell to be coloured
##            size - size of maze
##            colourVal - the intensity of the colour.
##  colourCell basically highlights the given cell by painting it with the given colourVal. Care should be taken that
##  the function doesn't paint over the black walls and only paints the empty spaces. This function returns the image
##  with the painted cell.
def colourCell(img, level, cellnum, size, colourVal):
    ############################# Add your Code Here ################################
    breadth=img[1].size
    height=img[0].size

    centre=(height/2,breadth/2)

    cent_x,cent_y,R,Theta=getCent(img,level,cellnum)

    radial_cell=40
    angular_cell=360/cellsPerLevel[level]

   
    startAngle= Theta-angular_cell*0.5+6-level
    endAngle= Theta+angular_cell*0.5-1.00001
    in_rad=int(round(R-radial_cell/2+4))
    out_rad=int(round(R+radial_cell/2-3))

    if level==6:
        startAngle= Theta-angular_cell*0.5+0.5
        endAngle= Theta+angular_cell*0.5-0.5
        in_rad=int(round(R-radial_cell/2+4))
        out_rad=int(round(R+radial_cell/2-3))

    
    angle=0
    thickness=2

    for radius in range(in_rad,out_rad):
        axes=(radius,radius)
        cv2.ellipse(img,centre,axes,angle,startAngle,endAngle,colourVal,thickness)



    #################################################################################  
    return img

##  Function that accepts some arguments from user and returns the graph of the maze image.
def buildGraph( img,size ):      ## You can pass your own arguments in this space.
    graph = {}
    ############################# Add your Code Here ################################
    if size==1:
        level=4
    elif size==2:
        level=6
    i=0
    j=0
    graph.update({(i,j):findNeighbours(img,i,j,size)})
    
    i=1
    j=1
    for i in range(1, level+1):
        for j in range(1, cellsPerLevel[i]+1):
            graph.update({(i,j):findNeighbours(img,i,j,size)})
    

    #################################################################################
    return graph

##  Function accepts some arguments and returns the Start coordinates of the maze.
def findStartPoint( img,size ):     ## You can pass your own arguments in this space.
    ############################# Add your Code Here ################################
    breadth=img[1].size
    height=img[0].size

    cent=(height/2,breadth/2)

    if size==1:
        level=4
    elif size==2:
        level=6

    if level==4:
        for cellnum in range(1,25):
            cent_x,cent_y,R,Theta=getCent(img,level,cellnum)
            ext=tuple(radialTravel(R,Theta,40/2,img))
            if img[ext]!=0:
                start=(level,cellnum)
                break
    if level==6:
        for cellnum in range(1,49):
            cent_x,cent_y,R,Theta=getCent(img,level,cellnum)
            ext=tuple(radialTravel(R,Theta,40/2,img))
            if img[ext]!=0:
                start=(level,cellnum)
                break
    
    

    #################################################################################
    return start

##  Finds shortest path between two coordinates in the maze. Returns a set of coordinates from initial point
##  to final point.
def findPath( maze_graph,start,end,path ):             ## You can pass your own arguments in this space.
    ############################# Add your Code Here ################################
    path=path+[start]
    if start == end:
        return path
    shortest = None
    for node in maze_graph[start]:
        if node not in path:
            newpath = findPath(maze_graph, node, end, path)
            if newpath:
                if not shortest or len(newpath) < len(shortest):
                     shortest = newpath 

    #################################################################################
    return shortest

## The findMarkers() function returns a list of coloured markers in form of a python dictionaries
## For example if a blue marker is present at (3,6) and red marker is present at (1,5) then the
## dictionary is returned as :-
##          list_of_markers = { 'Blue':(3,6), 'Red':(1,5)}
'''
*Function Name: findMarkers
*Input: imgHSV -> the image matrix in HSV format
*       size ->   the size of the image which determines the no of levels in the theta maze
*       imgBinary->the image matrix in binary colorspace
*Output: list_of_markers -> list of coloured markers in form of a python dictionary
*Logic: The centre of all the cells in the image are checked for their HSV values.If the values are found to be lying in a
        particular colour range , the cell position and the colour of the marker are added to the list.
*Example call -> listofMarkers=findMarkers(imgHSV,size,imgBinary)
'''

def findMarkers( imgHSV,size,imgBinary  ):             ## You can pass your own arguments in this space.
    list_of_markers = {}
    ############################# Add your Code Here ################################
    print size
    color=[120,10]
    for x in color:
        lower = np.array([x-10,50,50])
        upper = np.array([x+10,255,255])
        mask = cv2.inRange(imgHSV, lower, upper)
        #mask is a matrix with binary values for pixels which have the colour in the given range
        if size==1:
            for level in range(1,5):                                     #level changes its values for all levels of the theta matrix
                for cellnum in range(1,cellsPerLevel[level]+1):                #cellnum changes its values for all cells of a level 
                    cent_X,cent_Y,R,Theta=getCent(imgBinary,level,cellnum)     #cent_x,cent_y store the rectangular coordinates of the centre of the cell 
                    if (mask[cent_X,cent_Y]>0):                      #checking the value in 'mask' for all cells
                            if x==120:                               #checks for colour Blue
                                list_of_markers['Blue']=(level,cellnum)
                            elif x==10:                              #checks for colour red
                                list_of_markers['Red']=(level,cellnum)
        elif size==2:
            for level in range(1,7):
                for cellnum in range(1,cellsPerLevel[level]+1):
                    cent_X,cent_Y,R,Theta=getCent(imgBinary,level,cellnum)
                    if (mask[cent_X,cent_Y]>0):
                            if x==120:
                                list_of_markers['Blue']=(level,cellnum)
                            elif x==10:
                                list_of_markers['Red']=(level,cellnum)
                    
    #################################################################################
    return list_of_markers

## The findOptimumPath() function returns a python list which consists of all paths that need to be traversed
## in order to start from the START cell and traverse to any one of the markers ( either blue or red ) and then
## traverse to FINISH. The length of path should be shortest ( most optimal solution).
'''
*Function Name: findOptimumPath
*Input: imgBinary-> the image in binary format
*       listofMarkers -> the list of coloured markers obtained from the findMarkers function
*       size-> the size of the image which determines the no of levels in the theta maze
*Output:pathArray-> this contains a list of the cells in serial order that need to be traversed to follow the optimum path
*Logic:The path distance between START and FINISH through each of the markers is calculated.
       The path which has a shorter length is the optimum path.In case of equal length the path with the shortest distance from START to the
       marker is the one with optimum path
       To calculate shortest path findPath function is used
*Example call: path=findOptimumPath(imgBinary,listofMarkers,size)
'''

def findOptimumPath(imgBinary,listofMarkers,size    ):     ## You can pass your own arguments in this space.
    ############################# Add your Code Here ################################
    path_through_blue=[]
    # variable path_through_blue is the list that has path through blue marker 
    path_through_red=[]
    # variable path_through_red is the list that has path through red marker
    maze_graph = buildGraph( imgBinary,size  )   ## Build graph from maze image.
    start = findStartPoint( imgBinary,size )     ##findStartPoint returns the starting cell parameters
    path_through_blue.append(findPath(maze_graph,start,listofMarkers['Blue'],[]))  #path from START till Blue added
    path_through_blue.append(findPath(maze_graph,listofMarkers['Blue'],(0,0),[]))  #path from Blue to FINISH added
    path_through_red.append(findPath(maze_graph,start,listofMarkers['Red'],[]))    #path from START till red added
    path_through_red.append(findPath(maze_graph,listofMarkers['Red'],(0,0),[]))    #path from RED to FINISH added
    if len(path_through_blue)<len(path_through_red):
        pathArray=path_through_blue
    elif len(path_through_blue)>len(path_through_red):
        pathArray=path_through_red
    else:                        #conditon for equal path length through the both markers
        path_till_blue=findPath(maze_graph,start,listofMarkers['Blue'],[])      #path_till_blue has the path list cells from START till Blue marker
        path_till_red=findPath(maze_graph,start,listofMarkers['Red'],[])        #path_till_Red has the path list cells from START till Red marker
        if len(path_till_blue)>len(path_till_red):
            pathArray=path_through_red
        else:
            pathArray=path_through_blue
            


    #################################################################################
    return pathArray

## The colourPath() function highlights the whole path that needs to be traversed in the maze image and
## returns the final image.
'''
*Function Name:colourPath
*Input: imgBinary-> the image in binary format
*       path-> the list of cells for the optimum path obtained from the findOptimumPath function
*       size-> the size of the image which determines the no of levels in the theta maze
*Output:img-> final image of the theta matrix with the path highlighted with different colour
*Logic:Each and every in the path list is visted and colour is filled in the cell
*Example call: img=colourPath(imgBinary,size,path)
'''
def colourPath( imgBinary,size,path   ):   ## You can pass your own arguments in this space. 
    ############################# Add your Code Here ################################
    for subpath in path:   #subpath varible for a unique path within the complete path
        for cell in subpath:  #variable cell for the cell traversed in the path
            img=colourCell(imgBinary,cell[0],cell[1],size,220)

    #################################################################################
    return img

#####################################    Add Utility Functions Here   ###################################
##                                                                                                     ##
##                   You are free to define any functions you want in this space.                      ##
##                             The functions should be properly explained.                             ##




##                                                                                                     ##
##                                                                                                     ##
#########################################################################################################

## This is the main() function for the code, you are not allowed to change any statements in this part of
## the code. You are only allowed to change the arguments supplied in the findMarkers(), findOptimumPath()
## and colourPath() functions.    
def main(filePath, flag = 0):
    img = readImageHSV(filePath)
    imgBinary = readImageBinary(filePath)
    if len(img) == 440:
        size = 1
    else:
        size = 2
    listofMarkers = findMarkers(img,size,imgBinary)
    path = findOptimumPath(imgBinary,listofMarkers,size)
    img = colourPath(imgBinary,size,path)
    print path
    print listofMarkers
    if __name__ == "__main__":                    
        return img
    else:
        if flag == 0:
            return path
        elif flag == 1:
            return str(listofMarkers) + "\n"
        else:
            return img
    
## The main() function is called here. Specify the filepath of image in the space given.
if __name__ == "__main__":
    filePath = "image_09.jpg"     ## File path for test image
    img = main(filePath)           ## Main function call
    cv2.imshow("image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
