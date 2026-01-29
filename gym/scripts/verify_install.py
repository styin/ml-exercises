#!/usr/bin/env python
"""
Verify that Gymnasium and MuJoCo are correctly installed.
"""

import sys


def check_package(package_name: str, import_name: str = None) -> bool:
    """Check if a package is installed and importable."""
    import_name = import_name or package_name
    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "unknown")
        print(f"  [OK] {package_name}: {version}")
        return True
    except Exception as e:
        error_type = type(e).__name__
        print(f"  [FAIL] {package_name}: {error_type} - {str(e)[:80]}")
        return False


def test_mujoco_env() -> bool:
    """Test that MuJoCo environments work correctly."""
    try:
        import gymnasium as gym
        
        # Try to create a simple MuJoCo environment
        env = gym.make("InvertedPendulum-v5")
        obs, info = env.reset()
        
        # Take a few random steps
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
        
        env.close()
        print("  [OK] MuJoCo environment test: PASSED")
        return True
    except Exception as e:
        print(f"  [FAIL] MuJoCo environment test: FAILED - {e}")
        return False


def list_mujoco_envs() -> None:
    """List all available MuJoCo environments."""
    import gymnasium as gym
    
    print("\n" + "=" * 60)
    print("Available MuJoCo Environments (v5 recommended):")
    print("=" * 60)
    
    mujoco_envs = [
        spec.id for spec in gym.envs.registry.values()
        if "mujoco" in str(spec.entry_point).lower() or 
           any(name in spec.id for name in [
               "Ant", "HalfCheetah", "Hopper", "Humanoid", "Walker2d",
               "Swimmer", "Reacher", "Pusher", "InvertedPendulum", "InvertedDoublePendulum"
           ])
    ]
    
    # Group by base name
    env_groups = {}
    for env_id in sorted(set(mujoco_envs)):
        base_name = env_id.rsplit("-", 1)[0]
        if base_name not in env_groups:
            env_groups[base_name] = []
        env_groups[base_name].append(env_id)
    
    for base_name, versions in sorted(env_groups.items()):
        print(f"\n{base_name}:")
        for v in sorted(versions):
            print(f"    {v}")


def main():
    print("=" * 60)
    print("Gymnasium + MuJoCo Installation Verification")
    print("=" * 60)
    
    print("\n1. Checking required packages...")
    all_ok = True
    
    all_ok &= check_package("gymnasium")
    all_ok &= check_package("mujoco")
    all_ok &= check_package("numpy")
    
    print("\n2. Checking optional packages...")
    check_package("stable_baselines3", "stable_baselines3")
    check_package("tensorboard")
    check_package("imageio")
    check_package("matplotlib")
    
    print("\n3. Testing MuJoCo environment...")
    all_ok &= test_mujoco_env()
    
    # List available environments
    list_mujoco_envs()
    
    print("\n" + "=" * 60)
    if all_ok:
        print("[OK] All checks passed! Your installation is ready.")
        print("=" * 60)
        return 0
    else:
        print("[FAIL] Some checks failed. Please review the errors above.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
