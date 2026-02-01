import numpy as np
import gymnasium as gym
import pybullet as p
import pybullet_data
import os

# Evaluation settings
NUM_EVAL_EPISODES = 5
MAX_STEPS = 200

def load_policy(path):
    """Load policy and return weights"""
    return np.load(path)

def eval_mujoco(weights, num_episodes=NUM_EVAL_EPISODES):
    """Evaluate policy in MuJoCo Ant"""
    env = gym.make("Ant-v4")
    obs_dim = 27
    act_dim = 8
    
    # Pad or trim weights to match obs_dim
    if weights.shape[1] != obs_dim:
        W = np.zeros((act_dim, obs_dim))
        min_dim = min(weights.shape[1], obs_dim)
        W[:, :min_dim] = weights[:, :min_dim]
    else:
        W = weights
    
    total_reward = 0
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < MAX_STEPS:
            action = np.tanh(W @ obs[:obs_dim])
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1
    env.close()
    return total_reward / num_episodes

def eval_pybullet(weights, num_episodes=NUM_EVAL_EPISODES):
    """Evaluate policy in PyBullet Ant"""
    obs_dim = 28
    act_dim = 8
    
    # Pad or trim weights to match obs_dim
    if weights.shape[1] != obs_dim:
        W = np.zeros((act_dim, obs_dim))
        min_dim = min(weights.shape[1], obs_dim)
        W[:, :min_dim] = weights[:, :min_dim]
    else:
        W = weights
    
    total_reward = 0
    for _ in range(num_episodes):
        physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))
        ant = p.loadMJCF("/home/ec2-user/es_ablation/ant.xml")[0]
        num_joints = p.getNumJoints(ant)
        
        ep_reward = 0
        for step in range(MAX_STEPS):
            # Get observation
            pos, orn = p.getBasePositionAndOrientation(ant)
            vel, ang_vel = p.getBaseVelocity(ant)
            joints = []
            for i in range(min(num_joints, 8)):
                js = p.getJointState(ant, i)
                joints.extend([js[0], js[1]])
            obs = list(pos) + list(orn) + list(vel) + list(ang_vel) + joints
            obs = obs[:obs_dim] if len(obs) >= obs_dim else obs + [0]*(obs_dim-len(obs))
            obs = np.array(obs, dtype=np.float32)
            
            # Get action
            action = np.tanh(W @ obs)
            
            # Apply action
            for i in range(min(len(action), num_joints)):
                p.setJointMotorControl2(ant, i, p.VELOCITY_CONTROL, targetVelocity=float(action[i])*10)
            p.stepSimulation()
            
            # Reward
            new_pos, _ = p.getBasePositionAndOrientation(ant)
            reward = new_pos[0] - 0.001 * np.sum(np.square(action))
            ep_reward += reward
            
            if new_pos[2] < 0.2:
                break
        
        p.disconnect()
        total_reward += ep_reward
    
    return total_reward / num_episodes

def main():
    print("=" * 60)
    print("CROSS-SIMULATOR EVALUATION - ES ABLATION")
    print("=" * 60)
    
    # Collect all policies
    policies = {
        "mujoco_es": [],
        "pybullet_es": [],
        "isaac_es": []
    }
    
    for i in range(10):
        policies["mujoco_es"].append(load_policy(f"mujoco_es_policies/ant_mujoco_es_{i}.npy"))
        policies["pybullet_es"].append(load_policy(f"pybullet_es_policies/ant_pybullet_es_{i}.npy"))
        policies["isaac_es"].append(load_policy(f"isaac_policies/ant_isaac_{i}.npy"))
    
    # Results matrix
    results = {}
    
    for source in ["mujoco_es", "pybullet_es", "isaac_es"]:
        results[source] = {"mujoco": [], "pybullet": []}
        print(f"\nEvaluating {source} policies...")
        
        for i, policy in enumerate(policies[source]):
            print(f"  Policy {i}:", end=" ")
            
            # Eval on MuJoCo
            mj_score = eval_mujoco(policy)
            results[source]["mujoco"].append(mj_score)
            print(f"MuJoCo={mj_score:.1f}", end=" ")
            
            # Eval on PyBullet
            pb_score = eval_pybullet(policy)
            results[source]["pybullet"].append(pb_score)
            print(f"PyBullet={pb_score:.1f}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRANSFER MATRIX (mean ± std)")
    print("=" * 60)
    print(f"{'Source':<15} {'-> MuJoCo':<20} {'-> PyBullet':<20}")
    print("-" * 55)
    
    for source in ["mujoco_es", "pybullet_es", "isaac_es"]:
        mj = results[source]["mujoco"]
        pb = results[source]["pybullet"]
        print(f"{source:<15} {np.mean(mj):>7.1f} ± {np.std(mj):<7.1f} {np.mean(pb):>7.1f} ± {np.std(pb):<7.1f}")
    
    # Save results
    np.save("cross_eval_results.npy", results)
    print("\nResults saved to cross_eval_results.npy")

if __name__ == "__main__":
    main()
