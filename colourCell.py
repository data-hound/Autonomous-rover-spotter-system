def colourCell(img, level, cellnum, size, colourVal):
    ############################# Add your Code Here ################################
    breadth=img[1].size
    height=img[0].size

    centre=(height/2,breadth/2)

    cent_x,cent_y,R,Theta=getCent(img,level,cellnum)

    radial_cell=40
    angular_cell=360/cellsPerLevel[level]

    startAngle= Theta-angular_cell/2+1.00001
    endAngle= Theta+angular_cell/2-1.00001
    in_rad=int(round(R-radial_cell/2+4))
    out_rad=int(round(R+radial_cell/2-3))

    
    angle=0
    thickness=2

    for radius in range(in_rad,out_rad):
        axes=(radius,radius)
        cv2.ellipse(img,centre,axes,angle,startAngle,endAngle,colourVal,thickness)

    

    '''pointer=radialTravel(R,Theta,-radial_cell/2,img)
    pointer=rectToPol(img,pointer[0],pointer[1])
    print '1pointer=',pointer

    pointer=angularTravel(pointer[0],pointer[1],-angular_cell/2,img)
    pointer=pointer_=rectToPol(img,pointer[0],pointer[1])

    for i in range(pointer_[1],pointer_[1]+angular_cell):
        for j in range(pointer_[0],pointer_[0]+radial_cell):'''

    '''while pointer[0]<=pointer_[0]+radial_cell:
        pointer=radialTravel(pointer[0],pointer[1],1,img)
        pointer=rectToPol(img,pointer[0],pointer[1])
        print '2pointer=',pointer
        angular_step=(angular_cell/(pointer[0]*cosine(angular_cell)))
        while pointer[1]<=pointer_[1]+angular_cell:
            img[(int)(pointer[0])][(int)(pointer[1])]=colourVal
            pointer=angularTravel(pointer[0],pointer[1],angular_step,img)
            print '3pointer=',pointer
            pointer=rectToPol(img,pointer[0],pointer[1])
            if (int)(pointer[0])==0:
                break
            angular_step=angular_step+1
            print '4pointer=',pointer'''

        

    '''cv2.imshow("image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''


    #################################################################################
    return img
