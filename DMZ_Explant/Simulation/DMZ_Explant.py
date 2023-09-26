
from cc3d import CompuCellSetup
        

from DMZ_ExplantSteppables import DMZ_ExplantSteppable

CompuCellSetup.register_steppable(steppable=DMZ_ExplantSteppable(frequency=1))

from DMZ_ExplantSteppables import LeadingEdgeSteppable
CompuCellSetup.register_steppable(steppable=LeadingEdgeSteppable(frequency=1))
      
from DMZ_ExplantSteppables import ActinRingSteppable
CompuCellSetup.register_steppable(steppable=ActinRingSteppable(frequency=1))
        
from DMZ_ExplantSteppables import PassiveSteppable
CompuCellSetup.register_steppable(steppable=PassiveSteppable(frequency=1))
        
from DMZ_ExplantSteppables import SubstrateSteppable
CompuCellSetup.register_steppable(steppable=SubstrateSteppable(frequency=1))

CompuCellSetup.run()
