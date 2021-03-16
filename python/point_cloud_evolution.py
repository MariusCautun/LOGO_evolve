import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


def periodic_boundaries(pos,box):
    """Implements periodic boundaries."""
    for i in range(2):
        _s = pos[:,i] < 0
        if _s.sum() > 0: pos[:,i][_s] += box[i]
        _s = pos[:,i] >= box[i]
        if _s.sum() > 0: pos[:,i][_s] -= box[i]
    return


def initial_point_cloud(box,step=0.02,vel_sigma=0.2):
    """Initializes a uniform point distribution inside the box and assigns random velocities with the requested velocity disperssion."""
    _box = np.array(box)
    numX, numY = np.int(_box[0]/step)+1, np.int(_box[1]/step)+1
    print("Number of points:\n\t x-axis =", numX, "\n\t y-axis =", numY, "\n\t total  =", numX*numY)
    
    _x = np.linspace(0.,_box[0],num=numX)
    _y = np.linspace(0.,_box[1],num=numY)
    pos0 = np.empty( (numX*numY,2), np.float32 )
    pos0[:,0] = np.tile( _x, numY )
    pos0[:,1] = np.repeat( _y, numX )
    
    vel0 = np.random.normal(0,vel_sigma,size=(numX*numY,2)) * _box
    
    return pos0, vel0


def set_the_stage(box):
    plt.xlim( [0.,box[0]] )
    plt.ylim( [0.,box[1]] )
    plt.axis('off')
    return


def evolve_text( camera, points_pos, points_vel, spine_pos, spine_width, box, timestep=0.1, numTimeSteps=60, viscosityFactor=2., velocityFactor=0.93, plot_settings=set_the_stage ):
    pos = points_pos.copy()
    vel = points_vel.copy()
    dt = timestep
    
    from scipy.spatial import cKDTree
    tree = cKDTree( spine_pos )
    
    for j in range(numTimeSteps):
        dd, ii = tree.query(pos, k=1)
        dp = pos - spine_pos[ii]

        acc = -dp
        _s = dd < spine_width[ii]
        if _s.sum() > 0:
            acc[_s] = -viscosityFactor * vel[_s]

        pos += vel * dt + 0.5 * acc * dt**2
        vel += acc * dt
        vel *= .93

        periodic_boundaries( pos, box )
        plot_settings( box )
        
        plt.scatter( pos[:,0], pos[:,1], marker='.',s=10, c='w')
        plt.tight_layout()
        camera.snap()
        
    return camera, pos, vel


def implode_points(camera, points_pos, points_vel, center, box, numPeriods = 0.5, numIterations_perPeriod = 40, plot_settings=set_the_stage):
    numIterations = np.int(numPeriods * numIterations_perPeriod)
    step = 2./numIterations_perPeriod

    for j in range(numIterations+1):
        dp = points_pos - center
        pos = center + dp * np.abs(1.-j*step)

        periodic_boundaries( pos, box )

        plot_settings( box )
        plt.scatter( pos[:,0], pos[:,1], marker='.',s=10, c='w')
        plt.tight_layout()
        camera.snap()
    
    return camera, pos, points_vel


def explode_points(camera, points_pos, points_vel, box, timestep=0.1, numTimeSteps=20, plot_settings=set_the_stage):
    pos = points_pos.copy()
    vel = points_vel.copy()
    dt = timestep

    for j in range(numTimeSteps):
        pos += vel * dt

        periodic_boundaries(pos,box)

        plot_settings( box )
        plt.scatter( pos[:,0], pos[:,1], marker='.',s=10, c='w')
        plt.tight_layout()
        camera.snap()
    
    return camera, pos, vel


def read_target_skeleton(fileName):
    import pandas as pd

    target = pd.read_csv(fileName,delimiter=";",names=["x","y"],header=0,decimal=",")
    spine = np.array( target )

    xMin, yMin = spine.min(axis=0)
    xMax, yMax = spine.max(axis=0)
    width = 0.1 * np.ones( spine.shape[0] )

    vert_factor = 0.5
    text_height = (yMax-yMin) * vert_factor
    text_length = (xMax-xMin) * vert_factor

    spine[:] *= vert_factor
    width[:] *= vert_factor
    spine[:] += 0.25

    box_Lx = text_length + .5
    box_Ly = 1.
    box = np.array( [box_Lx, box_Ly] )
    
    return spine, width, box
