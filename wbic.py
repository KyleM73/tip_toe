import datetime
import numpy as np
import os
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client
import scipy as sp
import time
import yaml

class Env:
    def __init__(self, viz=True, cfg=None):
        self.date = datetime.datetime.now().strftime("%m%d_%H%M")
        self.viz = viz
        if isinstance(cfg,str):
            assert os.path.exists(cfg), "specified configuration file does not exist"
            with open(cfg, 'r') as stream:
                self.cfg = yaml.safe_load(stream)
        else:
            raise AssertionError("no configuration file specified")

        self.rng = np.random.default_rng(seed=self.cfg["SEED"])

        if not self.viz:
            self.client = bullet_client.BulletClient(connection_mode=p.DIRECT)
        else:
            self.client = bullet_client.BulletClient(connection_mode=p.GUI)
            self.client.configureDebugVisualizer(p.COV_ENABLE_GUI,0)


        ## initiate simulation
        self.client.resetSimulation()
        self.client.setTimeStep(1./self.cfg["PHYSICS_HZ"])
        self.client.setGravity(0, 0, -9.8)
        self.client.setPhysicsEngineParameter(enableFileCaching=0)

        ## setup ground
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.ground = self.client.loadURDF("plane.urdf")

        ## setup robot
        self.robot = p.loadURDF("robots/go1.urdf", [0, 0, 0.5], [0, 0, 0, 1], useFixedBase=False)

    def step(self):
        p.stepSimulation()

class MPC:
    def __init__(self):
        pass

def get_skew_sym_mat(v):
    assert v.size == 3
    v = np.ravel(v)
    return np.array([
        [0,-v[2],v[1]],
        [v[2],0,-v[0]],
        [-v[1],v[0],0]
        ])

def rot(yaw):
    return np.array([
        [np.cos(yaw), np.sin(yaw), 0],
        [-np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
        ])

def get_ref_com(yaw, x, y, z, yaw_rate_cmd, vx_cmd, vy_cmd, dt, k):
    A = np.block([
        [np.eye(3), np.zeros((3,3)), np.array([[0,0,0],[0,0,0],[0,0,dt/k]]), np.zeros((3,3)), np.zeros((3,1))],
        [np.zeros((3,3)), np.eye(3), np.zeros((3,3)), np.eye(3)*dt/k, np.zeros((3,1))],
        [np.zeros((3,3)), np.zeros((3,3)), np.eye(3), np.zeros((3,3)), np.zeros((3,1))],
        [np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), np.eye(3), np.zeros((3,1))], #np.array([[0],[0],[1]])
        [np.zeros((1,3)), np.zeros((1,3)), np.zeros((1,3)), np.zeros((1,3)), np.ones(1)]
        ])
    A_stack = np.vstack([np.linalg.matrix_power(A,i) for i in range(k+1)])
    x0 = np.array([[0,0,yaw,x,y,z,0,0,yaw_rate_cmd,vx_cmd,vy_cmd,0,-9.8]]).T
    return A_stack @ x0

yaw = 0
x,y,z = 1,0.5,0.5
yaw_cmd = 0.1
vx_cmd,vy_cmd = 1,0

traj = get_ref_com(yaw,x,y,z,yaw_cmd,vx_cmd,vy_cmd,1,1)
x0 = traj[:13]

def get_footstep_pose(v_cmd,w_cmd,x,shoulder,t_stance,k=0.03):
    # returns (4,3) array of footstep locations
    x = x.reshape(-1,1)
    ori = x[:3]
    pose = x[3:6]
    w = x[6:9]
    vel = x[9:12]
    g = x[12]

    p_shoulder = pose + rot(ori[2,0]) @ shoulder.T
    p_sym = t_stance/2 * vel.T + k*(vel.T - v_cmd)
    p_centrifugal = np.sqrt(np.abs(pose[2]/g))/2 * np.cross(vel.T,w_cmd)

    loc = p_shoulder.T + p_sym + p_centrifugal
    loc[:,2] = 0
    return loc

v_cmd = np.array([[vx_cmd,vy_cmd,0]])
w_cmd = np.array([[0,0,yaw_cmd]])
shoulder = x0[3:6].T + np.array([[0.25,0.25,0],[0.25,-0.25,0],[-0.25,0.25,0],[-0.25,-0.25,0]])
t_stance = 0.1

foot_pose = get_footstep_pose(v_cmd,w_cmd,x0,shoulder,t_stance)
print(foot_pose)

env = Env(True, "config.yaml")
for i in range(1000):
    env.step()
    time.sleep(1/1000.)