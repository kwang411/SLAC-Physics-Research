from larcv import larcv
from ROOT import TChain
import sys, os
import numpy as np

# Assumed constants: data product names
PARTICLE_LABEL='mcst'
PROJECTION0_LABEL='st_data_projection0'
PROJECTION1_LABEL='st_data_projection1'
# Instantiation of data product interface using ROOT TChain
exec('ch_part = TChain("particle_%s_tree")' % PARTICLE_LABEL)
exec('ch_img0 = TChain("image2d_%s_tree")'  % PROJECTION0_LABEL)
exec('ch_img1 = TChain("image2d_%s_tree")'  % PROJECTION1_LABEL)
exec('ch_clst = TChain("cluster3d_%s_tree")' % PARTICLE_LABEL)
chains=[ch_part,ch_img0,ch_img1,ch_clst]

# Read input arguments and sanity-check
data_file=None
entry=None
for argv in sys.argv:
    if argv.endswith('.root'):
        if not os.path.isfile(argv):
            print 'Input file not found',argv
            sys.exit(1)
        for ch in chains:
            ch.AddFile(argv)
            if not ch.GetEntries():
                print 'Input file does not contain necessary data...'
                sys.exit(1)
        data_file=argv
    elif argv.isdigit():
        entry=int(argv)
max_entry = chains[0].GetEntries()-1
if entry > max_entry:
    print 'Specified entry',entry,'is not found in the input file (max entry =',max_entry,')'
    sys.exit(1)

for ch in chains:
    ch.GetEntry(entry)

exec( 'data_part = ch_part.particle_%s_branch'  % PARTICLE_LABEL    )
exec( 'data_img0 = ch_img0.image2d_%s_branch'   % PROJECTION0_LABEL )
exec( 'data_img1 = ch_img1.image2d_%s_branch'   % PROJECTION1_LABEL )
exec( 'data_clst = ch_clst.cluster3d_%s_branch' % PARTICLE_LABEL    )

meta0  = data_img0.as_vector().front().meta()
meta1  = data_img1.as_vector().front().meta()
clst_v = data_clst.as_vector() 
parray = []
class pstruct:

    def __init__(self):
        self.track  = 0
        self.parent = 0
        self.pdg    = 0
        self.parent_pdg = 0
        self.energy = 0
        self.energy_dep = 0
        self.mom    = 0
        self.pos3d  = (0,0,0)
        self.dir3d  = (0,0,0)
        self.proj0  = (0,0)
        self.proj1  = (0,0)
        self.creation = ''
        self.num_vox = 0
        self.sum_vox = 0
        
    def __str__(self):
        msg  =''
        if self.track == self.parent: msg = '\033[95mPrimary\033[00m '
        else: msg = '\033[93mSecondary\033[00m '
        msg += '(ID=%d): ' % self.track
        msg += 'PDG %d ... Energy %g MeV (deposited %g) ... Mom. %g MeV/c\n' % (self.pdg, self.energy, self.energy_dep, self.mom)
        msg += '  Parent (ID=%d) PDG %d ... Creation by "%s"\n' % (self.parent, self.parent_pdg, self.creation)
        msg += '  Position  3D = (%g,%g,%g)\n' % self.pos3d
        msg += '  Direction 3D = (%g,%g,%g)\n' % self.dir3d
        msg += '  Number of 3D Voxels = %d ... sum %g\n'  % (self.num_vox,self.sum_vox)
        msg += '  Vertex on projection 0 pixel coordinate (%d,%d) ' % self.proj0
        msg += '... unit dir (%g,%g)\n' % (self.dir3d[0],self.dir3d[1])
        msg += '  Vertex on projection 1 pixel coordinate (%d,%d) ' % self.proj1
        msg += '... unit dir (%g,%g)\n' % (self.dir3d[1],self.dir3d[2])
        return msg

#print data_part.as_vector().size()
for index in xrange(data_part.as_vector().size()):
    p = data_part.as_vector()[index]
    pdata = pstruct()
    pdata.track  = p.track_id()
    pdata.parent = p.parent_track_id()
    pdata.pdg    = p.pdg_code()
    pdata.parent_pdg = p.parent_pdg_code()
    pdata.energy = p.energy_init()
    pdata.energy_dep = p.energy_deposit()
    pdata.mom    = np.sqrt(np.power(p.px(),2)+np.power(p.py(),2)+np.power(p.pz(),2))
    pdata.pos3d  = (p.position().x(),p.position().y(),p.position().z())
    pdata.dir3d  = (p.px() / pdata.mom, p.py() / pdata.mom, p.pz() / pdata.mom)
    pdata.proj0  = (meta0.col(p.position().x()), meta0.row(p.position().y()))
    pdata.proj1  = (meta1.col(p.position().y()), meta1.row(p.position().z()))
    pdata.creation = p.creation_process()
    pdata.num_vox = clst_v[index].as_vector().size()
    pdata.sum_vox = clst_v[index].sum()
    parray.append(pdata)

print
for pdata in parray:
    if not pdata.track == pdata.parent:
        continue
    print pdata

for pdata in parray:
    if pdata.track == pdata.parent:
        continue
    print pdata

    
    
    

