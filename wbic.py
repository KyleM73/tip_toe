import datetime
import numpy as np
import os
import osqp
#import osqp_c
import pinocchio as pin
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

        ## initialize simulation
        self.g = -9.8
        self.client.resetSimulation()
        self.client.setTimeStep(1./self.cfg["PHYSICS_HZ"])
        self.client.setGravity(0, 0, self.g)
        self.client.setPhysicsEngineParameter(enableFileCaching=0)

        ## setup ground
        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.ground = self.client.loadURDF("plane.urdf")

        ## setup robot
        self.robot = self.client.loadURDF(self.cfg["ROBOT_URDF"], [0, 0, 0.5], [0, 0, 0, 1], useFixedBase=False)

        ## let robot settle
        for _ in range(self.cfg["PHYSICS_HZ"]):
            self.client.stepSimulation()

        self.nJoints = p.getNumJoints(self.robot)
        self.joint_name2idx = {}
        self.active_joint_name2idx = {}
        self.link_name2idx = {}
        for i in range(self.nJoints):
          jointInfo = p.getJointInfo(self.robot, i)
          self.joint_name2idx[jointInfo[1].decode('UTF-8')] = jointInfo[0]
          self.link_name2idx[jointInfo[12].decode('UTF-8')] = jointInfo[0]
          if jointInfo[2] != p.JOINT_FIXED:
            self.active_joint_name2idx[jointInfo[1].decode('UTF-8')] = jointInfo[0]
        self.hip_name2idx = {k:v for k,v in self.link_name2idx.items() if k[-3:] == "hip"}
        self.hips = [k for k in self.hip_name2idx.keys()]
        self.feet = [k for k in self.link_name2idx.keys() if "foot" in k]
        self.active_joint_idx = [v for v in self.active_joint_name2idx.values()]

        ## initialize controller
        self.controller = Controller(self.cfg["ROBOT_URDF"])
        self.controller.update_kinematics(*self.get_q())
        self.controller.set_feet_links(self.feet)

    def step(self):
        x = self.get_obs()
        t_stance = np.array([[0.5,0.5,0.5,0.5]]).T
        links = self.client.getLinkStates(self.robot, [self.hip_name2idx[hip_name] for hip_name in self.hips])
        hip_poses = np.array([l[0] for l in links])
        footstep = self.controller.get_footstep_pose(x, hip_poses, t_stance)
        u = self.controller.step_MPC(x, hip_poses)
        for _ in range(self.cfg["PHYSICS_HZ"]//self.cfg["MPC_HZ"]):
            self.controller.update_kinematics(*self.get_q())
            torques = self.controller.step_WBC(u)
            torque = np.concatenate(torques)
            print(torque)
            self.client.setJointMotorControlArray(self.robot,
                self.active_joint_idx, p.TORQUE_CONTROL,
                forces=torque)
            self.client.stepSimulation()

    def get_obs(self):
        pose, oriq = self.client.getBasePositionAndOrientation(self.robot)
        ori = self.client.getEulerFromQuaternion(oriq)
        vel, avel = self.client.getBaseVelocity(self.robot)
        x = list(ori + pose + avel + vel)
        x.append(self.g)
        x[6:9] = self.controller.rot(x[2]) @ x[6:9] #convert to body frame
        return np.array(x).reshape(-1,1)

    def get_q(self):
        configuration = self.client.getJointStates(self.robot, self.active_joint_idx)
        q = [c[0] for c in configuration]
        q_dot = [c[1] for c in configuration]
        return np.array(q), np.array(q_dot)

class Controller:
    def __init__(self, robot_urdf):
        self.robot = pin.buildModelFromUrdf(robot_urdf)
        self.data = self.robot.createData()
        #for name, function in self.robot.__class__.__dict__.items():
        #    print(' **** %s: %s' % (name, function.__doc__))
        self.n = 4
        self.ns = 13
        self.dt = 0.5
        self.k = 10
        self.delta_t = self.dt/self.k
        self.m = 12
        self.mu = 0.6
        self.body_inertia = np.array([
            [0.13,0,0],
            [0,0.40,0],
            [0,0,0.45]
            ])
        self.indicator = np.array([[0],[0],[1]]) #gravity only acts in z direction
        self.z_weight = 50
        self.force_weight = 1e-6
        self.fmin = 0
        self.fmax = 200
        self.qp_problem = None
        self.Uz_last = 100 * np.ones((self.n*self.k,))
        self.c = np.array([[1/self.mu,0,0],[0,1/self.mu,0],[0,0,1]])
        self.foot_jacobian_selector = [1, 0, 3, 2] #picks which submatrix of jacobian goes with each foot

    def update_kinematics(self, q, q_dot):
        self.q, self.q_dot = q, q_dot
        pin.forwardKinematics(self.robot, self.data, q, q_dot)
        pin.updateFramePlacements(self.robot, self.data)

    def step_MPC(self, x, hips, mpc_update=True, yaw_cmd=0, vx_cmd=0, vy_cmd=0):
        x = x.reshape(-1,1)
        ori = x[:3]
        pose = x[3:6]
        w = x[6:9]
        vel = x[9:12]
        g = x[12]
        x_ref = self.get_ref_com(
            ori[2,0], pose[0,0], pose[1,0], pose[2,0],
            yaw_cmd, vx_cmd, vy_cmd
            )
        t_stance = self.get_stance_time()
        r = self.get_footstep_pose(x, hips, t_stance) #desired footstep location
        self.R = self.rot(ori[2,0]).T
        A = np.block([
            [np.eye(3), np.zeros((3,3)), self.R*self.delta_t, np.zeros((3,3)), np.zeros((3,1))],
            [np.zeros((3,3)), np.eye(3), np.zeros((3,3)), np.eye(3)*self.delta_t, np.zeros((3,1))],
            [np.zeros((3,3)), np.zeros((3,3)), np.eye(3), np.zeros((3,3)), np.zeros((3,1))],
            [np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), np.eye(3), self.indicator*self.delta_t], #np.array([[0],[0],[1]])
            [np.zeros((1,3)), np.zeros((1,3)), np.zeros((1,3)), np.zeros((1,3)), np.ones(1)]
            ])
        gI_inv = np.linalg.inv(self.R@self.body_inertia@self.R.T)
        B = np.block([
            [np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3))],
            [np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3))],
            [gI_inv@self.get_skew_sym_mat(r[i])*self.delta_t for i in range(self.n)],
            [np.eye(3)*self.delta_t/self.m for _ in range(self.n)],
            [np.zeros((1,3)) for _ in range(self.n)]
            ])
        U = self.MPC(A, B, x, x_ref)
        u = U[0,...]
        return u

    def step_WBC(self, u):
        return self.WBC_ground_force_control(u)

    def get_stance_time(self):
        return np.array([[0.5,0.5,0.5,0.5]]).T

    def get_ref_com(self, yaw, x, y, z, yaw_rate_cmd, vx_cmd, vy_cmd):
        A = np.block([
            [np.eye(3), np.zeros((3,3)), np.array([[0,0,0],[0,0,0],[0,0,self.delta_t]]), np.zeros((3,3)), np.zeros((3,1))],
            [np.zeros((3,3)), np.eye(3), np.zeros((3,3)), np.eye(3)*self.delta_t, np.zeros((3,1))],
            [np.zeros((3,3)), np.zeros((3,3)), np.eye(3), np.zeros((3,3)), np.zeros((3,1))],
            [np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), np.eye(3), np.zeros((3,1))], #np.array([[0],[0],[1]])
            [np.zeros((1,3)), np.zeros((1,3)), np.zeros((1,3)), np.zeros((1,3)), np.ones(1)]
            ])
        A_stack = np.vstack([np.linalg.matrix_power(A,i) for i in range(1, self.k+1)])
        x0 = np.array([[0,0,yaw,x,y,z,0,0,yaw_rate_cmd,vx_cmd,vy_cmd,0,-9.8]]).T
        return A_stack @ x0

    def pow(self, A, i):
        if i < 0:
            return np.zeros_like(A)
        elif i == 0:
            return np.eye(A.shape[-1])
        return np.power(A, i)


    def MPC(self, A, B, x0, x_ref):
        Aqp = np.block([
            [np.power(A, i)]
            for i in range(1, self.k+1)
            ])
        Bqp = np.zeros((13*self.k, 3*self.n*self.k))
        for row_order in range(1, self.k+1):
            Bqp[13*(row_order-1):13*row_order,:3*self.n*row_order] = np.block([self.pow(A,i-1)@B for i in range(row_order,0,-1)])

        L = np.eye(13*self.k)
        for i in range(self.k):
            L[5+13*i,5+13*i] = self.z_weight
        K = self.force_weight * np.eye(3*self.n*self.k)

        H = 2*(Bqp.T @ L @ Bqp + K)
        H = sp.sparse.csc_matrix(H)
        g = 2*Bqp.T @ L @ (Aqp @ x0 - x_ref)

        Uz_last_low = np.minimum(-self.Uz_last, 0)
        c_l = [np.array([Uz_last_low[i], Uz_last_low[i], self.fmin]) for i in range(self.n*self.k)]
        c_low = np.block(c_l).reshape(-1, 1)
        
        Uz_last_high = np.maximum(self.Uz_last, 0)
        c_h = [np.array([Uz_last_high[i], Uz_last_high[i], self.fmax]) for i in range(self.n*self.k)]
        c_high = np.block(c_h).reshape(-1, 1)

        C = sp.linalg.block_diag(*[self.c for _ in range(self.n*self.k)])
        C = sp.sparse.csc_matrix(C)

        if self.qp_problem is None:
            self.qp_problem = osqp.OSQP()
            self.H_first = H
            self.qp_problem.setup(H, g, C, c_low, c_high, warm_start=True, verbose=False)
            self.qp_problem.codegen("c_code", python_ext_name="osqp_c", parameters="matrices", force_rewrite=True)
            import osqp_c
        else:
            import osqp_c
            osqp_c.update_lin_cost(g)
            osqp_c.update_bounds(c_low, c_high)
            #print(H[H!=0])
            #assert False
            #osqp_c.update_P(H[H!=0], None, 0)
            #osqp_c.update_A(C[C!=0], None, 0)

            #self.qp_problem.update(q=g, l=c_low, u=c_high)
            #self.qp_problem.update(Px=np.array(H))
            #self.qp_problem.update(Ax=np.array(C))

        #result = self.qp_problem.solve()
        result = osqp_c.solve()
        x, y, status_val, iters, run_time = result

        if status_val > 1:
            print("Status: ",status_val)
            print("Iters: ",iters)
            raise ValueError("OSQP did not solve the problem!")

        U = x #[120,]
        U = U.reshape(self.k, self.n, 3)
        Uz = U.reshape(-1,3)
        self.Uz_last = Uz[:,2]
        return U

    def WBC_ground_force_control(self, grf):
        pin.computeJointJacobians(self.robot, self.data, self.q)
        T = []
        for i in range(len(self.feet_links)):
            frame_id = self.robot.getFrameId(self.feet_links[i])
            Jacobian = pin.getFrameJacobian(self.robot, self.data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            j = self.foot_jacobian_selector[i]
            J = Jacobian[0:3,3*j:3*(j+1)]
            T.append(J.T @ self.R.T @ grf[j,...])
        return T

    def set_feet_links(self, feet_links):
        self.feet_links = feet_links

    def get_skew_sym_mat(self, v):
        assert v.size == 3
        v = np.ravel(v)
        return np.array([
            [0,-v[2],v[1]],
            [v[2],0,-v[0]],
            [-v[1],v[0],0]
            ])

    def rot(self, yaw):
        return np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
            ])

    def get_footstep_pose(self, x, hips, t_stance):
        # returns (4,3) array of footstep locations
        x = x.reshape(-1,1)
        ori = x[:3]
        pose = x[3:6]
        w = x[6:9]
        vel = x[9:12]
        g = x[12]

        p_ref = pose + self.rot(ori[2,0]) @ hips.T
        p_des = p_ref.T + vel.T*t_stance/2
        p_des[:,2] = 0
        return p_des

    def get_footstep_pose_WBIC(self, v_cmd, w_cmd, x, hips, t_stance, k=0.03):
        # returns (4,3) array of footstep locations
        x = x.reshape(-1,1)
        ori = x[:3]
        pose = x[3:6]
        w = x[6:9]
        vel = x[9:12]
        g = x[12]

        p_hip = pose + self.rot(ori[2,0]) @ hips.T
        p_sym = vel.T*t_stance/2 + k*(vel.T - v_cmd)
        p_centrifugal = np.sqrt(np.abs(pose[2]/g))/2 * np.cross(vel.T,w_cmd)

        loc = p_hip.T + p_sym + p_centrifugal
        loc[:,2] = 0
        return loc


"""
yaw = 0
x,y,z = 1,0.5,0.5
yaw_cmd = 0.1
vx_cmd,vy_cmd = 1,0

traj = get_ref_com(yaw,x,y,z,yaw_cmd,vx_cmd,vy_cmd,1,1)
x0 = traj[:13]


v_cmd = np.array([[vx_cmd,vy_cmd,0]])
w_cmd = np.array([[0,0,yaw_cmd]])
shoulder = x0[3:6].T + np.array([[0.25,0.25,0],[0.25,-0.25,0],[-0.25,0.25,0],[-0.25,-0.25,0]])
t_stance = 0.1

foot_pose = get_footstep_pose(v_cmd,w_cmd,x0,shoulder,t_stance)
print(foot_pose)
"""

env = Env(True, "config.yaml")
for i in range(1000):
    env.step()
    time.sleep(1/20.)