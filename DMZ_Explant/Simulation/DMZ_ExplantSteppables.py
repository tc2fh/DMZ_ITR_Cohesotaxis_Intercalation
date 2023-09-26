from cc3d.core.PySteppables import *
from cc3d import CompuCellSetup
from cc3d.CompuCellSetup import persistent_globals as pg
import numpy as np
import random
from numpy import random as rng
import os


rng = np.random.default_rng()


timestep = 5 #1 mcs = 5 seconds if tissue moving at 0.25 pixels/10mcs

#Tissue Parameters
TissueLambda = 600
TTarget_distance = 5
Tmax_distance = 15

#Lamelipodia Parameters
LamellipodiaLambda = 800
LamellipodiaDistance = 2

LLTargetDist = 1
LLMaxDist = 15

#Passive Parameters
ifPassiveSubstrate = 1
SLink_TargetDist = 1
SLinkMaxDist = 5
SLinkLambda = 10

SL_CyclingRate = 1/360
SubLinkRate = timestep * SL_CyclingRate
# SubLinkDecay = .1

# Poisson
L_CyclingRate = 1/180
LamellaeRate = timestep * L_CyclingRate #reference DMZ Explant Rates for calibration data


# Cohesotaxis
ifCohesotaxis = 0 #Cohesotaxis by biasing pixels
Sigma = 6
ReverseBoolean = False

ifPythonCall = 0 #1 if running headless from python script so that we can read variables as passed in through that script, 0 if running from CC3D GUI
ifDataSave = 0 #1 if saving data, 0 if not saving data

plotflag = False


#Function to return probability weights mapped to sigmoid function
def SigWeights(x,b):
    '''
    Parameters
    ----------
    x : Integer value. Represents size of a vector to create weights for.
    b : Integer value. Represents the sigma parameter. This defines the slope
    of the probability by defining where the sigmoid function is in regards to
    the viewing window [-5,5] that we map the vector of probability weights to

    Returns
    -------
    weights : should return a vector of length 'x' that sum to 1
    representing a descending discrete probability distribution function
    mapped to the shape of the sigmoid within a defined viewing window

    '''
    #create a viewing window between [-5,5] of length x
    xrange = np.linspace(-5,5,x) 
    
    #Create a sigmoid function within that viewing window.
    # 'b' parameter will shift the sigmoid function left and right
    sigmoid = 1/(1+np.e**-(xrange-b))
    #normalize weights by dividing all values by sum of the vector
    weights = sigmoid/np.sum(sigmoid)
    weights = weights[::-1]
    return weights

#calculates cumulative distance of Pixel from all pixels in PixList
def PixelDist(Pixel, PixList):
    '''
    Parameters
    ----------
    Pixel : boundary_pixel_tracker_data Object.
    PixList : List of boundary_pixel_tracker_data objects.

    Returns
    -------
    CumDist: cumulative distance of Pixel to all pixels in PixList
    as calculated by pythagorean theorem
    '''
    CumDist=0 #cumulative distance
    for pixdata in PixList: #loop through pixels in PixList and do pythagorean to find distance
        CumDist+= np.sqrt((Pixel.pixel.x - pixdata.pixel.x)**2 + \
        (Pixel.pixel.y - pixdata.pixel.y)**2 + \
        (Pixel.pixel.z - pixdata.pixel.z)**2)

    return CumDist

class DMZ_ExplantSteppable(SteppableBasePy):

    def __init__(self,frequency=1):

        SteppableBasePy.__init__(self,frequency)

    def start(self):
        if ifPythonCall:
            values2pass = pg.input_object
            #Tissue Params
            global TissueLambda
            global TTarget_distance
            global Tmax_distance

            #Passive Cell-Substrate Params
            global SLinkLambda
            global SL_CyclingRate
            global SLink_TargetDist
            global SLinkMaxDist

            #lamellipodial parameters
            global LamellipodiaLambda
            global L_CyclingRate
            global LamellipodiaDistance
            global LLTargetDist
            global LLMaxDist
            global Sigma

            global RunNumber
            global OutputPath
            
            #added filename to values2pass input 5/9/2022
            global GeneratedFileName

            #Tissue Params
            TissueLambda = float(values2pass[0])
            TTarget_distance = float(values2pass[1])
            Tmax_distance = float(values2pass[2])

            #Passive Cell-Substrate Params
            SLinkLambda = float(values2pass[3])
            SL_CyclingRate = float(values2pass[4])
            SLink_TargetDist = float(values2pass[5])
            SLinkMaxDist = float(values2pass[6])

            #lamellipodial parameters
            LamellipodiaLambda = float(values2pass[7])
            L_CyclingRate = float(values2pass[8])
            LamellipodiaDistance = float(values2pass[9])
            LLTargetDist = float(values2pass[10])
            LLMaxDist = float(values2pass[11])
            Sigma = float(values2pass[12])

            RunNumber = float(values2pass[13])
            OutputPath = values2pass[14]
            
            #added filename to values2pass input 5/9/2022
            GeneratedFileName = values2pass[15]



        #CREATE SUBSTRATE
        SubstrateDim = 1
        a = np.arange(0.0,self.dim.x - SubstrateDim,SubstrateDim)
        b = np.arange(0.0,self.dim.y - SubstrateDim,SubstrateDim)
        for i in range(0,a.size):
            for z in range(0,b.size):
                x = a[i]
                y = b[z]
                cell = self.new_cell(self.SUBSTRATE)
                self.cell_field[x:(x + SubstrateDim), y:(y + SubstrateDim), 0] = cell
        return

    def step(self,mcs):
        # if mcs == 100:
            # self.resize_and_shift_lattice(new_size=(200, 100, 50), shift_vec=(0, 0, 0))
            # print("SHIFTED")
        return

    def finish(self):
        return

    def on_stop(self):
        return
class LeadingEdgeSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        

    def start(self):
        # CREATE LEADING EDGE LINKS
        # leading cell neighbors
        for cell in self.cell_list_by_type(self.LEADING):
            cell.dict['SubstrateLinkCounter'] = 0
            cell.dict['link'] = None
            #Create Leading edge links to neighbors
            for neighbor, _ in self.get_cell_neighbor_data_list(cell):
                    if neighbor and neighbor.type == self.PASSIVE and neighbor is not None and self.get_fpp_link_by_cells(cell,neighbor) is None:
                        link = self.new_fpp_link(cell, neighbor, TissueLambda, TTarget_distance, Tmax_distance)
                    elif neighbor and neighbor.type != self.SUBSTRATE:
                        self.new_fpp_link(cell, neighbor, TissueLambda, TTarget_distance, Tmax_distance)
        
        #Create initial lamellipodia link:
            #Find new substrate to link from cell boundary pixel:
            #Select a random pixel next to the medium
            FreePixelList = []
            CellAdhesionPixelList = []
            pixel_list = self.get_cell_boundary_pixel_list(cell)
            for boundary_pixel_tracker_data in pixel_list:
                if boundary_pixel_tracker_data.pixel.z == 1:
                    # this iterates over the list of boundary pixels, code modified from Jim Sluka's support forum post: https://www.reddit.com/r/CompuCell3D/comments/ewstcf/can_i_get_some_help_on_how_to_use_the_get_pixel/fg4qh6a/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button
                    (x1,y1,z1)=(boundary_pixel_tracker_data.pixel.x,
                                boundary_pixel_tracker_data.pixel.y,
                                boundary_pixel_tracker_data.pixel.z)
                    # iterate over 1st order neighbors of pixel of cell_1 at (x1,y1,z1)
                    # extend this list if you want 3D or higher order neighbors
                    for (dx,dy,dz) in ((1,0,0),(0,1,0),(-1,0,0),(0,-1,0)):  
                        # ifCell is the cell at the adjacent pixel
                        ifCell = self.cell_field[x1+dx,y1+dy,z1+dz] 
                        if not ifCell:  # if true it is not a cell, it is medium
                            FreePixelList.append(boundary_pixel_tracker_data)
                        else:
                            CellAdhesionPixelList.append(boundary_pixel_tracker_data)

            if len(FreePixelList) > 0:
                if ifCohesotaxis:
                    FreePixelList.sort(reverse=False, key=lambda pix: PixelDist(pix,CellAdhesionPixelList)) #Sort FreePixelList using PixelDist
                    weights = SigWeights(len(FreePixelList),Sigma) #Generate PDF of selection probability from weights function
                    test = [PixelDist(pix,CellAdhesionPixelList) for pix in FreePixelList]
                    # print('Pixel Distance= ',test)
                    # print('weights= ',weights)
                    SelectedBoundaryPix = rng.choice(FreePixelList,p=weights) #Select boundary pixel

                    
                else:
                    SelectedBoundaryPix = FreePixelList[np.random.randint(len(FreePixelList))]

                #find new substrate cell LamellipodiaDistance away from boundary in the appropriate direction
                DeltaX = SelectedBoundaryPix.pixel.x - cell.xCOM
                DeltaY = SelectedBoundaryPix.pixel.y - cell.yCOM
                Xfraction = DeltaX/(np.sqrt(DeltaX**2 + DeltaY**2))
                Yfraction = DeltaY/(np.sqrt(DeltaX**2 + DeltaY**2))
                Xcomponent = LamellipodiaDistance * Xfraction
                Ycomponent = LamellipodiaDistance * Yfraction
                CoordinateX = SelectedBoundaryPix.pixel.x + Xcomponent
                CoordinateY = SelectedBoundaryPix.pixel.y + Ycomponent
                NewLinkCell = self.cell_field[CoordinateX, CoordinateY, 0]

                self.new_fpp_link(cell, NewLinkCell, LamellipodiaLambda, LLTargetDist, LLMaxDist)
                cell.dict['link'] = NewLinkCell.id
                cell.dict['LinkTime'] = 0
        return

    def step(self, mcs):

        for cell in self.cell_list_by_type(self.LEADING):
            # if 'link' in cell.dict and mcs - cell.dict['LinkTime'] > LamellaeTime:
            if 'link' in cell.dict:
                #delete link
                PoissonProbability = 1-np.exp(-(LamellaeRate))
                if np.random.uniform() < PoissonProbability:
                    a = self.get_fpp_link_by_cells(cell, self.fetch_cell_by_id(cell.dict.pop('link')))
                    if a is not None:
                        self.delete_fpp_link(a)

            if 'link' not in cell.dict:
                #Find new substrate to link from cell boundary pixel:
                #Select a random pixel next to the medium
                FreePixelList = []
                CellAdhesionPixelList = []
                pixel_list = self.get_cell_boundary_pixel_list(cell)
                for boundary_pixel_tracker_data in pixel_list:
                    if boundary_pixel_tracker_data.pixel.z == 1:
                        # this iterates over the list of boundary pixels
                        (x1,y1,z1)=(boundary_pixel_tracker_data.pixel.x,
                                    boundary_pixel_tracker_data.pixel.y,
                                    boundary_pixel_tracker_data.pixel.z)
                        # iterate over 1st order neighbors of pixel of cell_1 at (x1,y1,z1)
                        # extend this list if you want 3D or higher order neighbors
                        for (dx,dy,dz) in ((1,0,0),(0,1,0),(-1,0,0),(0,-1,0)):  
                            # ifCell is the cell at the adjacent pixel
                            ifCell = self.cell_field[x1+dx,y1+dy,z1+dz] 
                            if not ifCell:  # if true it is not a cell, it is medium
                                # this neighbor pixel is medium so append original pixel
                                FreePixelList.append(boundary_pixel_tracker_data)
                            else:
                                CellAdhesionPixelList.append(boundary_pixel_tracker_data)
                
                if len(FreePixelList) > 0:
                    if ifCohesotaxis:
                        FreePixelList.sort(reverse=ReverseBoolean, key=lambda pix: PixelDist(pix,CellAdhesionPixelList)) #Sort FreePixelList using PixelDist
                        weights = SigWeights(len(FreePixelList),Sigma) #Generate PDF of selection probability from weights function
                        test = [PixelDist(pix,CellAdhesionPixelList) for pix in FreePixelList]
                        # print('Pixel Distance= ',test)
                        # print('weights= ',weights)
                        SelectedBoundaryPix = rng.choice(FreePixelList,p=weights) #Select boundary pixel

                    else:
                        SelectedBoundaryPix = FreePixelList[np.random.randint(len(FreePixelList))]

                    #find new substrate cell LamellipodiaDistance away from boundary in the appropriate direction
                    DeltaX = SelectedBoundaryPix.pixel.x - cell.xCOM
                    DeltaY = SelectedBoundaryPix.pixel.y - cell.yCOM
                    Xfraction = DeltaX/(np.sqrt(DeltaX**2 + DeltaY**2))
                    Yfraction = DeltaY/(np.sqrt(DeltaX**2 + DeltaY**2))
                    Xcomponent = LamellipodiaDistance * Xfraction
                    Ycomponent = LamellipodiaDistance * Yfraction
                    CoordinateX = SelectedBoundaryPix.pixel.x + Xcomponent
                    CoordinateY = SelectedBoundaryPix.pixel.y + Ycomponent
                    NewLinkCell = self.cell_field[CoordinateX, CoordinateY, 0]

                    if cell is None or NewLinkCell is None:
                        raise RuntimeError('Attempted link creation with the medium:', cell, NewLinkCell)

                    self.new_fpp_link(cell, NewLinkCell, LamellipodiaLambda, LLTargetDist, LLMaxDist)
                    cell.dict['link'] = NewLinkCell.id
                    cell.dict['LinkTime'] = mcs    

        return

    def finish(self):
        return

    def on_stop(self):
        return        
class ActinRingSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        

    def start(self):

        return
    def step(self, mcs):
        return

    def finish(self):
        return

    def on_stop(self):
        return
class PassiveSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        # passive neighbors
        for cell in self.cell_list_by_type(self.PASSIVE):
            cell.dict['SubstrateLinkCounter'] = 0
            for neighbor, _ in self.get_cell_neighbor_data_list(cell):
                    if neighbor and neighbor.type != self.SUBSTRATE and neighbor is not None and  self.get_fpp_link_by_cells(cell,neighbor) is None:
                        self.new_fpp_link(cell, neighbor, TissueLambda, TTarget_distance, Tmax_distance)
        return

    def step(self, mcs):
        if ifPassiveSubstrate:
            for cell in self.cell_list_by_type(self.PASSIVE):
                #create random links to substrate
                if 'link' not in cell.dict and cell.zCOM <= 4:
                    for neighbor, _ in self.get_cell_neighbor_data_list(cell):
                        if neighbor and neighbor.type == self.SUBSTRATE and cell.dict['SubstrateLinkCounter'] == 0:
                            NeighborDist = sqrt((neighbor.xCOM-cell.xCOM)**2 + (neighbor.yCOM-cell.yCOM)**2 + (neighbor.zCOM-cell.zCOM)**2)
                            if NeighborDist < 5:
                                self.new_fpp_link(cell, neighbor, SLinkLambda, SLink_TargetDist, SLinkMaxDist)
                                cell.dict['link'] = neighbor.id
                                cell.dict['SLinkTime'] = mcs
                                cell.dict['SubstrateLinkCounter'] += 1


                if 'link' in cell.dict and np.random.uniform() < (1-np.exp(-(SubLinkRate))) and cell.dict['SubstrateLinkCounter'] > 0:
                    a = self.get_fpp_link_by_cells(cell, self.fetch_cell_by_id(cell.dict.pop('link')))
                    if a is not None:
                        self.delete_fpp_link(a)
                        cell.dict['SubstrateLinkCounter'] -= 1

        return

    def finish(self):
        return

    def on_stop(self):
        return 
class SubstrateSteppable(SteppableBasePy): #Substrate currently has no behaviors, this steppable is currently used for saving data
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        if ifPythonCall:
            if ifDataSave:
                # fileDir = os.path.dirname(os.path.abspath(__file__))
                # fileDirUp = os.path.dirname(fileDir)

                filetitle = GeneratedFileName
                FileDir = OutputPath #+ r'\\'
                fileName = FileDir + filetitle
                

                self.file1 = open(fileName,"a")
                # self.file1.write("Start of the log file in the output directory\n")
                outputText = "mcs,Cell_ID,xposition,yposition,zposition\n"
                self.file1.write(outputText)
        return
    def step(self, mcs):
        if ifPythonCall:
            if ifDataSave:
                for cell in self.cell_list_by_type(self.LEADING):
                    outputText = str(mcs)+','+str(cell.id)+','+str(cell.xCOM)+','+str(cell.yCOM)+','+str(cell.zCOM)+'\n'
                    self.file1.write(outputText)
        return


    def finish(self):
        if ifPythonCall:
            if ifDataSave:
                self.file1.close()
        return

    def on_stop(self):
        if ifPythonCall:
            if ifDataSave:
                self.file1.close()
        return