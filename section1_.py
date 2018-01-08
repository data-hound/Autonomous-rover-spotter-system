import numpy as np
import cv2
import math
import time
cellsPerLevel=[1,6,12,24,24,24,48]
##  Returns sine of an angle.
def sine(angle):
    return math.sin(math.radians(angle))

##  Returns cosine of an angle
def cosine(angle):
    return math.cos(math.radians(angle))

##  Reads an image from the specified filepath and converts it to Grayscale. Then applies binary thresholding
##  to the image.
def readImage(filePath):
    mazeImg = cv2.imread(filePath)
    grayImg = cv2.cvtColor(mazeImg, cv2.COLOR_BGR2GRAY)
    ret,binaryImage = cv2.threshold(grayImg,127,255,cv2.THRESH_BINARY)
    return binaryImage

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

   
    startAngle= Theta-angular_cell*0.5+1.00001
    endAngle= Theta+angular_cell*0.5-1.00001
    in_rad=int(round(R-radial_cell/2+4))
    out_rad=int(round(R+radial_cell/2-3))

    if level==6:
        startAngle= Theta-angular_cell*0.5+0.5
        endAngle= Theta+angular_cell*0.5-0.5
        in_rad=int(round(R-radial_cell/2+4))
        out_rad=int(round(R+radial_cell/2-3))

    if level==2 :
        startAngle= Theta-angular_cell*0.5+2
        endAngle= Theta+angular_cell*0.5-2
        in_rad=int(round(R-radial_cell/2+4))
        out_rad=int(round(R+radial_cell/2-3))

    if level==1:
        startAngle= Theta-angular_cell*0.5+3.5
        endAngle= Theta+angular_cell*0.5-3.5
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
def buildGraph(img,size  ):   ## You can pass your own arguments in this space.
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
def findStartPoint(img,size):     ## You can pass your own arguments in this space.
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
                X1=0
                Y1=0
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
def findPath( maze_graph,start,end,path):      ## You can pass your own arguments in this space.
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

##  This is the main function where all other functions are called. It accepts filepath
##  of an image as input. You are not allowed to change any code in this function. You are
##  You are only allowed to change the parameters of the buildGraph, findStartPoint and findPath functions
def main(filePath, flag = 0):
    img = readImage(filePath)     ## Read image with specified filepath
    if len(img) == 440:           ## Dimensions of smaller maze image are 440x440
        size = 1
    else:
        size = 2
    maze_graph = buildGraph(img,size )   ## Build graph from maze image. Pass arguments as required
    start = findStartPoint(img,size )  ## Returns the coordinates of the start of the maze
    shortestPath = findPath(maze_graph,start,(0,0),[])  ## Find shortest path. Pass arguments as required.
    print shortestPath
    string = str(shortestPath) + "\n"
    for i in shortestPath:               ## Loop to paint the solution path.
        img = colourCell(img, i[0], i[1], size, 230)
    if __name__ == '__main__':     ## Return value for main() function.
        return img
    else:
        if flag == 0:
            return string
        else:
            return graph
## The main() function is called here. Specify the filepath of image in the space given.
if __name__ == "__main__":
    filepath = "image_08.jpg"     ## File path for test image
    img = main(filepath)          ## Main function call
    cv2.imshow("image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
